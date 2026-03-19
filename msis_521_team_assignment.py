# -*- coding: utf-8 -*-
"""
MSIS 521 Team Project — Restaurant Insight-to-Action Engine (Model/Pipeline)

Purpose
-------
Given a CSV of restaurant reviews, this script:
1) Sanitizes review text to reduce obvious PII risk (emails/phones/URLs/long IDs).
2) Builds a TF-IDF representation of the reviews (course: text representation).
3) Clusters reviews into themes using KMeans (course: unsupervised learning).
4) Produces two outputs for the Streamlit dashboard:
   - processed_reviews.csv (row-level assignments)
   - theme_summary.csv (theme-level insights + a strategy plan)

Notes
-----
- This pipeline is intentionally lightweight and explainable (demo-friendly).
- It does NOT require any paid APIs. Strategy text is rule-based so it runs offline.
- The Streamlit app calls this script via subprocess and expects the output filenames
  exactly as: processed_reviews.csv and theme_summary.csv in the provided --outdir.

CLI
---
python msis_521_team_assignment.py --input reviews.csv --outdir runs/20260101_120000
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------
# Configuration
# ----------------------------

PROMO_LIBRARY = {
    "value": [
        ("Bundle Deal", "Main + side + drink for a fixed price (Thu–Sun)."),
        ("Free Add-on", "Free fries or drink upgrade with any entrée (Thu–Sun)."),
        ("Portion Upgrade", "Upgrade to large for $1 (Thu–Sun)."),
        ("BOGO", "Buy one entrée, get the second 50% off (Fri–Sun)."),
    ],
    "speed": [
        ("Fast Lunch Promise", "10–15 min lunch promise (Mon–Fri lunch hours)."),
        ("Order-ahead Bonus", "10% off when ordering ahead (weekday)."),
    ],
    "service": [
        ("Service Recovery Card", "If we miss expectations, get a bounce-back coupon for next visit."),
    ],
    "cleanliness": [
        ("Fresh & Clean Week", "Highlight sanitation steps + small thank-you offer (e.g., free drink upgrade)."),
    ],
    "quality": [
        ("Chef’s Pick", "Limited-time feature of best-selling dish (Fri–Sun)."),
        ("Freshness Spotlight", "Ingredient spotlight + small add-on offer (Thu–Sun)."),
    ],
}


def categorize_theme(keywords: list[str]) -> str:
    k = {w.lower() for w in (keywords or [])}

    if k & {"expensive", "price", "pricing", "value", "cost", "worth", "cheap", "deal"}:
        return "value"
    if k & {"slow", "wait", "waiting", "line", "minutes", "time", "fast", "quick"}:
        return "speed"
    if k & {"rude", "friendly", "staff", "service", "manager", "cashier", "helpful"}:
        return "service"
    if k & {"dirty", "clean", "hygiene", "bathroom", "restroom"}:
        return "cleanliness"
    if k & {"delicious", "fresh", "taste", "tasty", "quality", "flavor", "spicy"}:
        return "quality"

    # safe default for restaurant marketing
    return "value"


PROCESSED_FILENAME = "processed_reviews.csv"
SUMMARY_FILENAME = "theme_summary.csv"

DEFAULT_N_CLUSTERS = 6
DEFAULT_MAX_FEATURES = 5000
DEFAULT_NGRAM_RANGE = (1, 2)

# Fallback actions when no specific theme-to-action match is found
DEFAULT_ACTIONS = [
    "- Highlight a best-selling item using a real customer quote; run Fri–Sun as a limited-time special.",
    "- Offer a simple bundle (main + side + drink) to improve perceived value; test pricing for 2 weeks.",
    "- Respond to <4-star reviews within 24 hours with a polite apology + invitation to return (service recovery).",
]

# ----------------------------
# Sanitization (preprocessing)
# ----------------------------

EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE_RE = re.compile(r"(\+?\d{1,2}[\s-]?)?(\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
LONG_ID_RE = re.compile(r"\b\d{6,}\b")  # long numeric identifiers


def sanitize_text(text: str) -> str:
    """Mask obvious PII patterns and normalize whitespace."""
    if pd.isna(text):
        return ""
    s = str(text)
    s = re.sub(r"\s+", " ", s).strip()
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = PHONE_RE.sub("[PHONE]", s)
    s = URL_RE.sub("[URL]", s)
    s = LONG_ID_RE.sub("[ID]", s)
    return s


def clean_for_vectorizer(text: str) -> str:
    """A minimal cleaning step for TF-IDF (keep it simple & explainable)."""
    s = sanitize_text(text).lower()
    # Keep letters/numbers/basic punctuation as separators
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------
# Strategy rules (offline)
# ----------------------------

@dataclass(frozen=True)
class EventRule:
    name: str
    keywords: Tuple[str, ...]
    suggestion: str


EVENT_RULES: List[EventRule] = [
    EventRule(
        name="Happy Hour Push",
        keywords=("drinks", "cocktail", "bar", "beer", "wine", "happy hour"),
        suggestion="Run a weekday happy hour promo (e.g., 4–6pm) and post a short reel featuring drinks + ambience.",
    ),
    EventRule(
        name="Lunch Combo Offer",
        keywords=("lunch", "quick", "fast", "work", "office", "downtown"),
        suggestion="Create a lunch combo (main + side + drink) and promote it with 'in-and-out' messaging.",
    ),
    EventRule(
        name="Family Bundle",
        keywords=("family", "kids", "children", "group", "birthday"),
        suggestion="Offer a family bundle (2 mains + 2 sides + dessert) and promote it for weekends.",
    ),
    EventRule(
        name="Portion / Value Upgrade",
        keywords=("portion", "small", "tiny", "value", "price", "expensive"),
        suggestion="Test a 'regular/large' size option or a small portion increase with a modest price adjustment.",
    ),
    EventRule(
        name="Service Recovery Playbook",
        keywords=("service", "rude", "slow", "wait", "waiting", "staff"),
        suggestion="Improve response time and add service recovery: apology + invite back + manager follow-up for low ratings.",
    ),
    EventRule(
        name="Ambience / Date Night",
        keywords=("ambience", "atmosphere", "romantic", "date", "music", "vibe"),
        suggestion="Run a date-night promotion (set menu / dessert pairing) and highlight ambience in posts.",
    ),
]


def suggest_events_from_keywords(keywords: list[str], top_k: int = 3) -> list[str]:
    bucket = categorize_theme(keywords)
    options = PROMO_LIBRARY.get(bucket, PROMO_LIBRARY["value"])

    # Take top_k options and format as bullet lines (important for your Streamlit parser)
    picks = options[:top_k]
    return [f"- {title}: {desc}" for title, desc in picks]


def make_weekly_calendar(theme_name: str, promo_weeks: int = 2, primary_promo: str = "") -> list[str]:
    """
    Outputs one line per week:
      Week N: Mon: ... | Thu: ...
    Includes the chosen promo title + mechanics.
    """

    def _parse_promo(p: str) -> tuple[str, str]:
        p = (p or "").strip()
        if not p:
            return ("Bundle Deal", "Main + side + drink for a fixed price (Thu–Sun).")

        p = re.sub(r"^\s*[-•*]\s*", "", p).strip()
        title, desc = p, ""
        if ":" in p:
            title, desc = p.split(":", 1)
        elif "—" in p:
            title, desc = p.split("—", 1)
        elif ";" in p:
            title, desc = p.split(";", 1)

        return (title.strip(), desc.strip())

    promo_title, promo_desc = _parse_promo(primary_promo)

    week_lines: list[str] = []
    for wk in range(1, int(promo_weeks) + 1):
        if wk == 1:
            mon = f"Highlight {theme_name} using 1 customer quote + photo"
            thu = f"Launch {promo_title} — {promo_desc} (CTA: Ask for the promo)"
        elif wk == 2:
            mon = f"Explain what’s included in {promo_title} (value message)"
            thu = f"Weekend reminder for {promo_title} + UGC: Tag us / leave a review"
        elif wk == 3:
            mon = "Behind-the-scenes / quality proof (short video/photo)"
            thu = f"Run {promo_title} again + small variation (limited-time / add-on)"
        elif wk == 4:
            mon = "Repost best customer photo/review from earlier weeks"
            thu = f"Final push for {promo_title} + review drive"
        else:
            # Weeks 5–8: structured repeat cycles
            cycle = (wk - 5) % 4
            if cycle == 0:
                mon = "Top picks list (3 items) tied to the theme"
                thu = f"Run {promo_title} with “best value this week” framing"
            elif cycle == 1:
                mon = "Social proof post: top customer favorites + rating"
                thu = f"Run {promo_title} + mini UGC contest (tag-to-win)"
            elif cycle == 2:
                mon = "Teaser post: “This weekend only”"
                thu = f"Scarcity push: {promo_title} (weekend only)"
            else:
                mon = "Thank-you post + what we improved / what customers loved"
                thu = f"Last chance: {promo_title} + follow/subscribe CTA"

        week_lines.append(f"Week {wk}: Mon: {mon} | Thu: {thu}")

    return week_lines


# ----------------------------
# Core pipeline
# ----------------------------

def load_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Required columns (as enforced by Streamlit)
    required = {"business_name", "text", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}. Need: business_name, text, rating")

    # Ensure types
    df["text"] = df["text"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    return df


def build_tfidf_and_cluster(
    texts: List[str],
    n_clusters: int = DEFAULT_N_CLUSTERS,
    max_features: int = DEFAULT_MAX_FEATURES,
    ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> Tuple[TfidfVectorizer, KMeans, np.ndarray]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
    )
    X = vectorizer.fit_transform(texts)

    # KMeans is stochastic; keep random_state fixed for stable demos
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    return vectorizer, kmeans, labels


def top_keywords_per_cluster(vectorizer: TfidfVectorizer, kmeans: KMeans, top_n: int = 12) -> Dict[int, List[str]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    out: Dict[int, List[str]] = {}
    for c in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[c]
        top_idx = center.argsort()[::-1][:top_n]
        out[c] = feature_names[top_idx].tolist()
    return out


def generate_theme_name(cluster_id: int, keywords: List[str]) -> str:
    """
    A simple, explainable theme name:
    "Theme 1 — service, staff, friendly"
    (1-based numbering so it matches the Streamlit display)
    """
    short = ", ".join(keywords[:3]) if keywords else "misc"
    return f"Theme {cluster_id + 1} — {short}"


def build_theme_summary(
    processed: pd.DataFrame,
    keywords_by_cluster: Dict[int, List[str]],
    promo_weeks: int = 2,
) -> pd.DataFrame:
    """Create a theme-level table for dashboard + strategy generation."""
    rows: list[dict] = []
    baseline = float(pd.to_numeric(processed.get("rating"), errors="coerce").mean())

    for cluster_id, kw in keywords_by_cluster.items():
        subset = processed[processed["theme_cluster"] == cluster_id]
        size = int(len(subset))
        avg_rating = float(pd.to_numeric(subset.get("rating"), errors="coerce").mean()) if size > 0 else float("nan")

        theme_name = generate_theme_name(cluster_id, kw)

        # Promo OPTIONS (3)
        actions = suggest_events_from_keywords(kw, top_k=3)

        # Choose the FIRST option as the primary promo for the weekly plan
        primary_promo = actions[0] if actions else "- Bundle Deal: Main + side + drink for a fixed price (Thu–Sun)."

        # Week-by-week plan embeds the chosen promo
        content_plan = make_weekly_calendar(
            theme_name=theme_name,
            promo_weeks=promo_weeks,
            primary_promo=primary_promo,
        )

        # Evidence quotes make the strategy defensible in a demo.
        examples = (
            subset.get("text_sanitized", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .str.strip()
        )
        examples = [t for t in examples.tolist() if t][:3]
        evidence_quotes = " || ".join(examples)

        # Fix vs Promote classification (simple, explainable heuristic)
        if not np.isnan(avg_rating) and avg_rating < (baseline - 0.15):
            theme_type = "Fix"
        else:
            theme_type = "Promote"

        priority_score = (size or 0) * (abs(avg_rating - baseline) if not np.isnan(avg_rating) else 0.0)

        fallback_actions = DEFAULT_ACTIONS
        ai_strategy = "\n".join(
            ["Recommended Actions (pick 1–2):"] + (actions if actions else fallback_actions)
            + ["", "Content Plan:"] + content_plan
        )

        rows.append(
            {
                "Theme_ID": cluster_id,
                "Theme_Label": theme_name,
                "Theme_Type": theme_type,
                "Review_Count": size,
                "Avg_Rating": round(avg_rating, 2) if not np.isnan(avg_rating) else np.nan,
                "Baseline_Rating": round(baseline, 2) if not np.isnan(baseline) else np.nan,
                "Top_Keywords": ", ".join(kw),
                "Evidence_Quotes": evidence_quotes,
                "Priority_Score": round(float(priority_score), 4),
                "AI_Strategy": ai_strategy,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        # Priority label from score quantiles (High/Medium/Low)
        q1 = out["Priority_Score"].quantile(0.33)
        q2 = out["Priority_Score"].quantile(0.66)

        def _prio(s: float) -> str:
            if s >= q2:
                return "High"
            if s >= q1:
                return "Medium"
            return "Low"

        out["Priority"] = out["Priority_Score"].apply(_prio)
        out = out.sort_values(["Priority_Score", "Review_Count"], ascending=[False, False]).reset_index(drop=True)

    return out


def run_pipeline(input_csv: Path, outdir: Path, n_clusters: int, promo_weeks: int) -> Tuple[Path, Path]:
    df = load_reviews(input_csv).copy()

    # Keep raw text for auditability in demo
    df["text_raw"] = df["text"].astype(str)
    df["text_sanitized"] = df["text_raw"].apply(sanitize_text)
    df["cleaned_text"] = df["text_raw"].apply(clean_for_vectorizer)

    # Cluster on cleaned_text
    vectorizer, kmeans, labels = build_tfidf_and_cluster(
        df["cleaned_text"].fillna("").tolist(),
        n_clusters=n_clusters,
    )
    df["theme_cluster"] = labels.astype(int)

    keywords_by_cluster = top_keywords_per_cluster(vectorizer, kmeans, top_n=12)

    # Attach keywords list per row
    df["Top_Keywords"] = df["theme_cluster"].map(lambda c: ", ".join(keywords_by_cluster.get(int(c), [])))

    summary = build_theme_summary(df, keywords_by_cluster, promo_weeks=promo_weeks)

    outdir.mkdir(parents=True, exist_ok=True)
    processed_path = outdir / PROCESSED_FILENAME
    summary_path = outdir / SUMMARY_FILENAME

    df.to_csv(processed_path, index=False)
    summary.to_csv(summary_path, index=False)

    return processed_path, summary_path


# ----------------------------
# CLI entry
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restaurant Insight-to-Action Engine pipeline")
    p.add_argument("--input", type=str, default="reviews.csv",
                   help="Input reviews CSV (must contain business_name, text, rating)")
    p.add_argument("--outdir", type=str, default=".",
                   help="Output directory for processed_reviews.csv and theme_summary.csv")
    p.add_argument("--clusters", type=int, default=DEFAULT_N_CLUSTERS,
                   help="Number of KMeans clusters (themes). Recommended: 5–8")
    p.add_argument("--promo_weeks", type=int, default=2,
                   help="Number of weeks to generate in the promo calendar.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    outdir = Path(args.outdir)

    processed_path, summary_path = run_pipeline(
        input_csv=input_csv,
        outdir=outdir,
        n_clusters=int(args.clusters),
        promo_weeks=int(args.promo_weeks),
    )

    # Print a short success message for Streamlit logs
    print(json.dumps({
        "status": "ok",
        "input": str(input_csv),
        "outdir": str(outdir),
        "processed": str(processed_path),
        "summary": str(summary_path),
        "clusters": int(args.clusters),
        "promo_weeks": int(args.promo_weeks),
    }))


if __name__ == "__main__":
    main()