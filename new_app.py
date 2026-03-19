# new_app.py
import io
import re
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# MUST be the first Streamlit call (and only once)
st.set_page_config(page_title="Restaurant Insight-to-Action Engine", layout="wide")

# ----------------------------
# Configuration
# ----------------------------
PROJECT_ROOT = Path(__file__).parent
ML_SCRIPT = PROJECT_ROOT / "msis_521_team_assignment.py"
RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_OUT = "processed_reviews.csv"
SUMMARY_OUT = "theme_summary.csv"

PAGES = ["1) Upload Reviews", "2) Analyze & Strategy", "3) Dashboard"]

# ----------------------------
# Theme (Solid Light Blue)
# ----------------------------
st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"]{
        background: #cfe8ff;
        color: #0b1f3a;
    }
    .block-container{
        background: #b7dcff;
        border: 1px solid rgba(11, 31, 58, 0.12);
        border-radius: 18px;
        padding: 1.6rem 1.6rem 2rem 1.6rem;
        box-shadow: 0 14px 30px rgba(11, 31, 58, 0.12);
    }
    html, body, [class*="css"], .stMarkdown, .stText, .stCaption, .stSubheader, .stHeader, .stTitle{
        color: #0b1f3a !important;
    }
    section[data-testid="stSidebar"]{
        background: #a9d4ff;
        border-right: 1px solid rgba(11, 31, 58, 0.12);
    }
    section[data-testid="stSidebar"] *{
        color: #0b1f3a !important;
    }
    .stButton > button{
        border-radius: 12px;
        padding: 0.55rem 1rem;
        border: 1px solid rgba(11, 31, 58, 0.18);
        background: #3b82f6;
        color: #ffffff;
        font-weight: 700;
    }
    .stButton > button:hover{
        background: #2563eb;
        border: 1px solid rgba(11, 31, 58, 0.22);
    }
    input, textarea{
        background: #d9efff !important;
        color: #0b1f3a !important;
        border: 1px solid rgba(11, 31, 58, 0.18) !important;
        border-radius: 10px !important;
    }

    /* Select + multiselect styling (and remove the focus "bar") */
    div[data-baseweb="select"] > div{
        background: #d9efff !important;
        color: #0b1f3a !important;
        border: 1px solid rgba(11, 31, 58, 0.18) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-baseweb="select"] > div:focus-within{
        box-shadow: none !important;
        outline: none !important;
    }

    details{
        background: #d9efff;
        border: 1px solid rgba(11, 31, 58, 0.14);
        border-radius: 14px;
        padding: 0.25rem 0.8rem;
    }
    details summary{
        color: #0b1f3a !important;
        font-weight: 700;
    }
    div[data-testid="stDataFrame"]{
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(11, 31, 58, 0.14);
        background: #d9efff;
    }
    .stAlert{ border-radius: 14px; }

    /* Hide the blinking text cursor inside select / multiselect inputs */
    div[data-baseweb="select"] input{
        caret-color: transparent !important;
    }

    /* Also hide any selection highlight inside those inputs (optional) */
    div[data-baseweb="select"] input::selection{
        background: transparent !important;
    }

    div[data-baseweb="select"] input:focus,
    div[data-baseweb="select"] input:focus-visible{
        outline: none !important;
        box-shadow: none !important;
    }

    /* Kill the blinking caret inside Streamlit/BaseWeb selects */
    div[data-baseweb="select"] input[role="combobox"]{
        caret-color: transparent !important;
    }

    /* Hide the extra cursor/overlay div that appears immediately after the combobox input */
    div[data-baseweb="select"] input[role="combobox"] + div{
        display: none !important;
    }

    /* (Optional) remove focus glow as well */
    div[data-baseweb="select"] input[role="combobox"]:focus,
    div[data-baseweb="select"] input[role="combobox"]:focus-visible{
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Restaurant Insight-to-Action Engine")

# ----------------------------
# Session defaults
# ----------------------------
for key, default in {
    "uploaded_bytes": None,
    "uploaded_name": None,
    "raw_df": None,
    "selected_businesses": [],
    "promo_weeks": None,
    "last_promo_weeks": None,
    "sample_n": 300,
    "run_dir": None,
    "processed_df": None,
    "summary_df": None,
    "strategy_table": None,
    "last_run_logs": None,
    "run_health": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------
# Cached CSV reader
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


# ----------------------------
# Table helper (hide index)
# ----------------------------
def show_table(df: pd.DataFrame, key: Optional[str] = None):
    """Display a dataframe without showing index column (fallback-safe)."""
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, key=key)
    except TypeError:
        st.dataframe(df.reset_index(drop=True), use_container_width=True, key=key)


# ----------------------------
# Navigation helpers
# ----------------------------
def _set_page(page_name: str):
    st.session_state["nav_page"] = page_name


def nav_bar(current_page: str):
    idx = PAGES.index(current_page)
    back_page = PAGES[idx - 1] if idx > 0 else None
    next_page = PAGES[idx + 1] if idx < len(PAGES) - 1 else PAGES[0]

    next_disabled = False
    if current_page == "1) Upload Reviews":
        next_disabled = st.session_state.get("uploaded_bytes") is None
    elif current_page == "2) Analyze & Strategy":
        next_disabled = st.session_state.get("summary_df") is None

    col_back, _, col_next = st.columns([1.2, 6, 1.2])
    with col_back:
        if back_page is not None:
            st.button("← Back", key=f"back_{current_page}", on_click=_set_page, args=(back_page,))
    with col_next:
    # Hide the Next button on the last page
        if current_page != "3) Dashboard":
            st.button(
                "Next →",
                key=f"next_{current_page}",
                disabled=next_disabled,
                on_click=_set_page,
                args=(next_page,),
            )


# ----------------------------
# ML runner helpers
# ----------------------------
DISPLAY_SHIM = """
# ---- injected by Streamlit runner (safe in plain Python) ----
try:
    from IPython.display import display  # type: ignore
except Exception:
    def display(x):
        print(x)
# ------------------------------------------------------------
"""


def create_sanitized_ml_copy(original_script: Path, dest_script: Path, input_csv_path: Path) -> None:
    """
    Creates a sanitized copy of the ML script that strips notebook/shell lines.
    Also strips common show() calls that pop browser windows.
    """
    text = original_script.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("/content/reviews.csv", str(input_csv_path).replace("\\", "/"))

    out_lines = [DISPLAY_SHIM]
    skip_contains = ("pip install", "streamlit run", "localtunnel", "ngrok")

    for ln in text.splitlines(True):
        s = ln.lstrip()

        if s.startswith("!"):
            out_lines.append("# [stripped notebook shell] " + ln)
            continue
        if s.startswith("%%") or s.startswith("%"):
            out_lines.append("# [stripped notebook magic] " + ln)
            continue
        if "get_ipython().system" in s:
            out_lines.append("# [stripped get_ipython system] " + ln)
            continue
        if any(tok in s for tok in skip_contains):
            out_lines.append("# [stripped shell/run] " + ln)
            continue

        if "plt.show" in s or "fig.show" in s or s.strip().endswith(".show()"):
            out_lines.append("# [stripped show()] " + ln)
            continue

        out_lines.append(ln)

    dest_script.write_text("".join(out_lines), encoding="utf-8")


def _outputs_exist(run_dir: Path) -> bool:
    return (run_dir / PROCESSED_OUT).exists() and (run_dir / SUMMARY_OUT).exists()


def run_ml_pipeline(script_path: Path, input_csv_path: Path, run_dir: Path, promo_weeks: Optional[int]) -> Dict:
    """
    Runs the ML pipeline via a sanitized copy.
    Passes --promo_weeks if supported; retries without if not supported.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"ML script not found: {script_path}")

    sanitized = run_dir / "ml_sanitized_run.py"
    create_sanitized_ml_copy(script_path, sanitized, input_csv_path)

    def _run(cmd_args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd_args, capture_output=True, text=True, cwd=str(run_dir), timeout=900)

    base_cmd = [
        sys.executable,
        str(sanitized),
        "--input",
        str(input_csv_path),
        "--outdir",
        str(run_dir),
    ]

    cmd1 = base_cmd.copy()
    if promo_weeks is not None:
        cmd1 += ["--promo_weeks", str(int(promo_weeks))]

    p = _run(cmd1)
    if p.returncode == 0 or _outputs_exist(run_dir):
        return {"mode": "sanitized.args", "cmd": cmd1, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "sanitized": str(sanitized)}

    # Retry without promo_weeks if unrecognized
    if promo_weeks is not None and "unrecognized arguments" in (p.stderr or "").lower() and "promo_weeks" in (p.stderr or "").lower():
        p_retry = _run(base_cmd)
        if p_retry.returncode == 0 or _outputs_exist(run_dir):
            return {"mode": "sanitized.retry_no_promo", "cmd": base_cmd, "returncode": p_retry.returncode, "stdout": p_retry.stdout, "stderr": p_retry.stderr, "sanitized": str(sanitized)}

    # Last fallback: run from cwd only
    p2 = _run([sys.executable, str(sanitized)])
    if p2.returncode == 0 or _outputs_exist(run_dir):
        return {"mode": "sanitized.cwd_only", "cmd": [sys.executable, str(sanitized)], "returncode": p2.returncode, "stdout": p2.stdout, "stderr": p2.stderr, "sanitized": str(sanitized)}

    raise RuntimeError(
        "Failed to run ML pipeline (sanitized copy).\n\n"
        f"Sanitized script: {sanitized}\n\n"
        f"Attempt 1 cmd: {cmd1}\nreturncode: {p.returncode}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}\n\n"
        f"Attempt 2 cmd: {[sys.executable, str(sanitized)]}\nreturncode: {p2.returncode}\nstdout:\n{p2.stdout}\nstderr:\n{p2.stderr}\n"
    )


# ----------------------------
# Optional course-friendly sanity check
# ----------------------------
def compute_silhouette_if_possible(processed_df: pd.DataFrame) -> Optional[float]:
    if "cleaned_text" not in processed_df.columns or "theme_cluster" not in processed_df.columns:
        return None

    texts = processed_df["cleaned_text"].fillna("").astype(str).tolist()
    labels = processed_df["theme_cluster"]

    if labels.nunique(dropna=True) < 2:
        return None

    vec = TfidfVectorizer(max_features=1000)
    X = vec.fit_transform(texts)

    if X.shape[0] > 600:
        X = X[:600]
        labels = labels.iloc[:600]

    try:
        return float(silhouette_score(X, labels.astype(int)))
    except Exception:
        return None


def _validate_required_cols(df: pd.DataFrame) -> List[str]:
    required = {"business_name", "text", "rating"}
    return sorted(list(required - set(df.columns)))


# ----------------------------
# Theme display + simplified labels
# ----------------------------
def _short_keywords_for_label(row: pd.Series, max_k: int = 4) -> str:
    """
    Returns "kw1, kw2, kw3, kw4" best-effort from:
      - Top_Keywords column (preferred)
      - or Theme_Label text like: "Theme 0 — expensive, delicious, good"
    """
    kw = str(row.get("Top_Keywords", "")).strip()

    if not kw:
        tl = str(row.get("Theme_Label", "")).strip()
        parts = re.split(r"\s*[—-]\s*", tl, maxsplit=1)
        if len(parts) > 1:
            kw = parts[1].strip()

    if not kw:
        return ""

    toks = [t.strip() for t in re.split(r"[,\|;]+", kw) if t.strip()]
    return ", ".join(toks[:max_k])


def _extract_keywords_from_row(row: pd.Series) -> List[str]:
    kw = str(row.get("Top_Keywords", "")).strip()
    if not kw:
        tl = str(row.get("Theme_Label", "")).strip()
        parts = re.split(r"\s*[—-]\s*", tl, maxsplit=1)
        if len(parts) > 1:
            kw = parts[1].strip()
    return [t.strip().lower() for t in re.split(r"[,\|;]+", kw) if t.strip()]


def simplify_theme_label(row: pd.Series) -> str:
    kws = _extract_keywords_from_row(row)

    generic = {"good", "great", "nice", "best", "ok", "okay", "love", "loved", "bad", "awesome", "amazing", "delicious"}
    kws_set = {k for k in kws if k not in generic}

    rules = [
        ("Pricing / Value", {"price", "prices", "expensive", "cheap", "value", "cost", "worth", "deal"}),
        ("Service / Staff", {"service", "staff", "server", "rude", "friendly", "manager", "cashier", "attitude", "helpful"}),
        ("Wait Time / Speed", {"wait", "waiting", "slow", "line", "quick", "fast", "minutes", "time", "speed"}),
        ("Cleanliness", {"clean", "dirty", "hygiene", "bathroom", "restroom"}),
        ("Food Quality / Taste", {"taste", "tasty", "fresh", "flavor", "quality", "seasoning", "bland", "salty", "spicy"}),
        ("Portion / Quantity", {"portion", "small", "large", "size", "quantity", "amount"}),
        ("Breakfast / Coffee", {"breakfast", "brunch", "coffee", "eggs", "morning", "latte", "espresso"}),
        ("Desserts / Treats", {"dessert", "treat", "treats", "cake", "cookie", "ice", "sweet", "brownie"}),
        ("Ambience / Venue", {"venue", "atmosphere", "music", "seating", "space", "place", "vibe", "ambience"}),
        ("Photos / Presentation", {"photo", "picture", "presentation", "plating"}),
    ]

    best_label = "General Experience"
    best_score = 0
    for label, vocab in rules:
        score = len(kws_set.intersection(vocab))
        if score > best_score:
            best_score = score
            best_label = label

    return best_label


def add_theme_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - Theme_Number: 1-based integer (Theme 1..N)
      - Theme_Display: "Theme 1 (kw1, kw2, kw3, kw4)"
      - Theme_Label_Simple: business-friendly label
    """
    s = summary_df.copy()

    # Stable ordering by cluster id if present
    if "Cluster" in s.columns:
        s = s.sort_values("Cluster", kind="stable").reset_index(drop=True)
        s["Theme_Number"] = s["Cluster"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)
    elif "Theme_ID" in s.columns:
        s = s.sort_values("Theme_ID", kind="stable").reset_index(drop=True)
        s["Theme_Number"] = s["Theme_ID"].apply(lambda x: int(x) + 1 if pd.notna(x) else None)
    else:
        s = s.reset_index(drop=True)
        s["Theme_Number"] = [i + 1 for i in range(len(s))]

    kw_short = s.apply(lambda r: _short_keywords_for_label(r, max_k=4), axis=1)
    s["Theme_Display"] = s["Theme_Number"].apply(lambda n: f"Theme {int(n)}" if pd.notna(n) else "Theme")
    mask = kw_short.astype(str).str.strip().ne("")
    s.loc[mask, "Theme_Display"] = (
        s.loc[mask, "Theme_Number"].apply(lambda n: f"Theme {int(n)}").values
        + " (" + kw_short.loc[mask].values + ")"
    )

    s["Theme_Label_Simple"] = s.apply(simplify_theme_label, axis=1)
    return s


# ----------------------------
# Fix/Promote helper
# ----------------------------
def add_fix_promote_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a simple business-facing column:
      - Fix_Promote: Fix First / Fix / Promote / Monitor
    Uses Avg_Rating + Review_Count (volume).
    """
    s = summary_df.copy()

    # Normalize Review_Count
    if "Review_Count" not in s.columns and "Size" in s.columns:
        s["Review_Count"] = s["Size"]
    s["Review_Count"] = pd.to_numeric(s.get("Review_Count", 0), errors="coerce").fillna(0).astype(int)

    # Normalize Avg_Rating
    s["Avg_Rating"] = pd.to_numeric(s.get("Avg_Rating", None), errors="coerce")

    # High-volume threshold = median (with a small floor)
    vol_threshold = int(s["Review_Count"].median()) if len(s) else 0
    vol_threshold = max(vol_threshold, 5)

    def _fix_promote(row) -> str:
        r = row["Avg_Rating"]
        v = row["Review_Count"]

        if pd.isna(r):
            return "Monitor"

        # Fix side
        if r <= 3.8 and v >= vol_threshold:
            return "Fix First"
        if r <= 3.8 and v < vol_threshold:
            return "Fix"

        # Promote side
        if r >= 4.3 and v >= vol_threshold:
            return "Promote"
        if r >= 4.3 and v < vol_threshold:
            return "Promote (Niche)"

        return "Monitor"

    s["Fix_Promote"] = s.apply(_fix_promote, axis=1)
    return s


# ----------------------------
# Strategy parsing + simplified week text
# ----------------------------
def _simplify_week_text(text: str, max_words: int = 26) -> str:
    """
    Keep BOTH Mon + Thu. Only remove long 'Details:' tail and truncate.
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if not t:
        return ""

    # Remove "Details: ...." (optional)
    t = re.sub(r"\s*Details:\s*.*$", "", t, flags=re.IGNORECASE).strip()

    # Normalize separators to make it readable
    t = t.replace("Mon—", "Mon: ").replace("Thu—", "Thu: ")
    t = t.replace("Mon–", "Mon: ").replace("Thu–", "Thu: ")
    t = t.replace("Mon-", "Mon: ").replace("Thu-", "Thu: ")

    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]) + "…"
    return t


def _parse_ai_strategy_to_columns(ai_text: str, promo_weeks: int) -> Dict[str, str]:
    """
    Parse AI_Strategy into:
      - Recommended_Action (only 1 action)
      - Content_Plan_Week_1..N (simplified week lines)
    """
    if not isinstance(ai_text, str):
        ai_text = ""

    lines = ai_text.splitlines()

    # Weeks (tolerant: ":" or "-" and optional bullets)
    week_map: Dict[int, str] = {}
    week_re = re.compile(r"^\s*(?:[-•*]\s*)?Week\s*(\d+)\s*[:\-–—]\s*(.+?)\s*$", re.IGNORECASE)
    for ln in lines:
        m = week_re.match(ln.strip())
        if m:
            week_map[int(m.group(1))] = _simplify_week_text(m.group(2).strip())

    # Actions section (tolerant headings + bullet markers)
    actions: List[str] = []
    in_actions = False
    for ln in lines:
        s = ln.strip()
        low = s.lower()

        if ("recommended actions" in low) or ("recommended promotional events" in low) or ("recommended promos" in low):
            in_actions = True
            continue

        if in_actions and (low.startswith("content plan") or "calendar" in low or low.startswith("week ")):
            in_actions = False

        if in_actions and re.match(r"^\s*[-•*]\s+", ln):
            item = re.sub(r"^\s*[-•*]\s+", "", ln).strip()
            if item:
                actions.append(item)

    out: Dict[str, str] = {}

    # Only one recommended action
    reco = actions[0] if actions else ""
    out["Recommended_Action"] = _simplify_week_text(reco, max_words=14) if reco else ""

    # Fill weeks 1..promo_weeks; if missing, use short defaults
    filler = "Repeat best offer + UGC"
    last_known: Optional[str] = None
    for wk in range(1, promo_weeks + 1):
        if wk in week_map and week_map[wk]:
            last_known = week_map[wk]
            out[f"Content_Plan_Week_{wk}"] = week_map[wk]
        else:
            out[f"Content_Plan_Week_{wk}"] = last_known if last_known else filler

    return out


def _best_effort_evidence_quotes(summary_df: pd.DataFrame, processed_df: pd.DataFrame) -> List[str]:
    """Attach up to 3 example reviews per theme (best-effort)."""
    text_col = None
    for c in ["text_redacted", "text_sanitized", "text", "review_text"]:
        if c in processed_df.columns:
            text_col = c
            break
    if text_col is None:
        return [""] * len(summary_df)

    if "Cluster" in summary_df.columns and "theme_cluster" in processed_df.columns:
        quotes = []
        for _, r in summary_df.iterrows():
            cid = r.get("Cluster")
            sub = processed_df[processed_df["theme_cluster"] == cid]
            ex = sub[text_col].fillna("").astype(str).str.strip().tolist()
            ex = [t for t in ex if t][:3]
            quotes.append(" || ".join(ex))
        return quotes

    if "Theme_ID" in summary_df.columns and "theme_cluster" in processed_df.columns:
        quotes = []
        for _, r in summary_df.iterrows():
            cid = r.get("Theme_ID")
            sub = processed_df[processed_df["theme_cluster"] == cid]
            ex = sub[text_col].fillna("").astype(str).str.strip().tolist()
            ex = [t for t in ex if t][:3]
            quotes.append(" || ".join(ex))
        return quotes

    return [""] * len(summary_df)


def build_strategy_table(summary_df: pd.DataFrame, processed_df: pd.DataFrame, selected_businesses: List[str], promo_weeks: int) -> pd.DataFrame:
    """
    Output requirements:
      - Theme numbering starts at 1 and shows keywords in brackets (Theme_Display)
      - Simplified label in separate column
      - Fix_Promote column present
      - Only 1 recommended action column
      - Week-wise content is simplified
      - NO Top_Keywords as a separate column in strategy preview/CSV
      - Remove Recommended_Action if empty across all rows
    """
    s = summary_df.copy()

    biz_label = "; ".join(selected_businesses) if selected_businesses else "ALL_BUSINESSES"
    s.insert(0, "Restaurant(s)", biz_label)

    # Normalize counts if needed
    if "Review_Count" not in s.columns and "Size" in s.columns:
        s["Review_Count"] = s["Size"]
    if "Review_Count" not in s.columns:
        s["Review_Count"] = ""
    if "Avg_Rating" not in s.columns:
        s["Avg_Rating"] = ""
    if "AI_Strategy" not in s.columns:
        s["AI_Strategy"] = ""

    # Ensure theme columns exist (in case caller didn't add them)
    if "Theme_Number" not in s.columns:
        s["Theme_Number"] = [i + 1 for i in range(len(s))]
    if "Theme_Display" not in s.columns:
        s["Theme_Display"] = s["Theme_Number"].apply(lambda n: f"Theme {int(n)}")
    if "Theme_Label_Simple" not in s.columns:
        s["Theme_Label_Simple"] = s.apply(simplify_theme_label, axis=1)

    # Fix/Promote
    if "Fix_Promote" not in s.columns:
        s = add_fix_promote_columns(s)

    # Evidence
    if "Evidence_Quotes" not in s.columns or s["Evidence_Quotes"].fillna("").astype(str).str.strip().eq("").all():
        s["Evidence_Quotes"] = _best_effort_evidence_quotes(s, processed_df)

    # Parse AI_Strategy
    parsed_rows = []
    for _, row in s.iterrows():
        parsed_rows.append(_parse_ai_strategy_to_columns(str(row.get("AI_Strategy", "")), promo_weeks))
    parsed_df = pd.DataFrame(parsed_rows)

    base_cols = [
        "Restaurant(s)",
        "Theme_Number",
        "Theme_Display",
        "Theme_Label_Simple",
        "Fix_Promote",
        "Review_Count",
        "Avg_Rating",
        "Evidence_Quotes",
    ]
    base_cols = [c for c in base_cols if c in s.columns]

    out = pd.concat([s[base_cols].reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

    # Drop Recommended_Action if all empty
    if "Recommended_Action" in out.columns:
        all_empty = out["Recommended_Action"].fillna("").astype(str).str.strip().eq("").all()
        if all_empty:
            out = out.drop(columns=["Recommended_Action"])

    week_cols = [f"Content_Plan_Week_{wk}" for wk in range(1, promo_weeks + 1)]
    ordered = base_cols + (["Recommended_Action"] if "Recommended_Action" in out.columns else []) + week_cols
    return out[[c for c in ordered if c in out.columns]]


# ----------------------------
# Upload callback
# ----------------------------
def on_upload_change():
    up = st.session_state.get("uploader_csv")
    if up is None:
        return

    st.session_state.uploaded_bytes = up.getvalue()
    st.session_state.uploaded_name = up.name

    st.session_state.raw_df = None
    st.session_state.processed_df = None
    st.session_state.summary_df = None
    st.session_state.strategy_table = None
    st.session_state.last_run_logs = None
    st.session_state.run_dir = None
    st.session_state.run_health = False

    st.session_state.selected_businesses = []
    st.session_state.promo_weeks = None
    st.session_state.last_promo_weeks = None


# ----------------------------
# Pages
# ----------------------------
def page_upload():
    nav_bar("1) Upload Reviews")
    st.header("Upload Reviews")

    st.file_uploader(
        "Upload reviews (CSV)",
        type=["csv"],
        key="uploader_csv",
        on_change=on_upload_change,
    )

    if st.session_state.uploaded_bytes is None:
        st.info("Upload a CSV to begin. (Next is enabled after upload.)")
        return

    try:
        raw = read_csv_cached(st.session_state.uploaded_bytes)
        st.session_state.raw_df = raw
    except Exception as e:
        st.error("Could not read the uploaded CSV.")
        st.exception(e)
        return

    st.caption(f"Loaded: {st.session_state.uploaded_name} — {raw.shape[0]} rows × {raw.shape[1]} columns")

    missing = _validate_required_cols(raw)
    if missing:
        st.error(f"Missing required columns: {missing} (need business_name, text, rating).")
        return

    st.subheader("Preview")
    show_table(raw.head(10), key="preview_raw")

    with st.expander("Data Health (optional)", expanded=False):
        st.write("Detected columns:", list(raw.columns))

        if st.button("Run Data Health checks", key="run_health_btn"):
            st.session_state.run_health = True

        if not st.session_state.run_health:
            st.info("Click 'Run Data Health checks' to compute metrics/charts.")
            return

        c1, c2, c3, c4, c5 = st.columns(5)

        total_rows = len(raw)
        uniq_rest = raw["business_name"].nunique(dropna=True)

        text_series = raw["text"]
        missing_text = int(text_series.isna().sum())
        empty_text = int(text_series.fillna("").astype(str).str.strip().eq("").sum())
        pct_bad_text = round(((missing_text + empty_text) / max(total_rows, 1)) * 100, 2)

        missing_rating = int(raw["rating"].isna().sum())
        pct_missing_rating = round((missing_rating / max(total_rows, 1)) * 100, 2)

        dup_count = int(raw.duplicated(subset=["business_name", "text", "rating"]).sum())

        c1.metric("Rows", total_rows)
        c2.metric("Restaurants", uniq_rest)
        c3.metric("Bad text %", f"{pct_bad_text}%")
        c4.metric("Missing rating %", f"{pct_missing_rating}%")
        c5.metric("Duplicates", dup_count)

        st.write("Rating distribution")
        try:
            tmp = raw.copy()
            tmp["rating_bin"] = pd.to_numeric(tmp["rating"], errors="coerce").round().astype("Int64")
            counts = tmp["rating_bin"].value_counts(dropna=True).sort_index()
            df_counts = counts.rename_axis("rating_bin").reset_index(name="count")
            fig_r = px.bar(df_counts, x="rating_bin", y="count", labels={"rating_bin": "Rating (rounded)", "count": "Count"})
            st.plotly_chart(fig_r, use_container_width=True)
        except Exception:
            st.caption("Rating distribution unavailable.")


def page_analyze_and_strategy():
    nav_bar("2) Analyze & Strategy")
    st.header("Analyze + Generate Strategy")

    if st.session_state.raw_df is None:
        if st.session_state.uploaded_bytes is None:
            st.warning("Upload a CSV on Page 1 first.")
            return
        st.session_state.raw_df = read_csv_cached(st.session_state.uploaded_bytes)

    raw = st.session_state.raw_df
    missing = _validate_required_cols(raw)
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    businesses = sorted(raw["business_name"].dropna().unique().tolist())

    st.subheader("Selections")
    selected_businesses = st.multiselect(
        "Choose restaurant(s) to analyze (1–3 recommended)",
        businesses,
        default=st.session_state.get("selected_businesses", []),
    )
    st.session_state.selected_businesses = selected_businesses

    promo_choice = st.selectbox("Number of weeks for promotions", ["Select…", 2, 4, 8], index=0, key="promo_choice")
    promo_weeks = None if promo_choice == "Select…" else int(promo_choice)
    st.session_state.promo_weeks = promo_weeks

    # If promo weeks changed, clear generated strategy table so it rebuilds correctly
    if st.session_state.get("last_promo_weeks") != promo_weeks:
        st.session_state.strategy_table = None
        st.session_state.last_promo_weeks = promo_weeks

    sample_n = st.slider("Max reviews to analyze (for speed)", 50, 600, st.session_state.get("sample_n", 300), 50)
    st.session_state.sample_n = sample_n

    st.divider()

    colA, colB = st.columns(2)
    run_clicked = colA.button("Sanitize + Analyze", disabled=(promo_weeks is None))
    strategy_clicked = colB.button("Generate Strategy", disabled=(promo_weeks is None or st.session_state.get("summary_df") is None))

    if promo_weeks is None:
        st.info("Select number of weeks to enable analysis.")

    if run_clicked:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        run_df = raw.copy()
        if selected_businesses:
            run_df = run_df[run_df["business_name"].isin(selected_businesses)]
        if len(run_df) > sample_n:
            run_df = run_df.sample(n=sample_n, random_state=42)

        input_csv_path = run_dir / "reviews.csv"
        run_df.to_csv(input_csv_path, index=False)

        with st.spinner("Running ML pipeline..."):
            logs = run_ml_pipeline(ML_SCRIPT, input_csv_path, run_dir, promo_weeks=promo_weeks)
            st.session_state.last_run_logs = logs

        processed_path = run_dir / PROCESSED_OUT
        summary_path = run_dir / SUMMARY_OUT
        if not processed_path.exists() or not summary_path.exists():
            st.error("Expected outputs not found. Check PROCESSED_OUT/SUMMARY_OUT filenames.")
            st.stop()

        st.session_state.run_dir = run_dir
        st.session_state.processed_df = pd.read_csv(processed_path)

        # Add Theme_Number/Theme_Display/Theme_Label_Simple + Fix_Promote
        tmp_summary = add_theme_columns(pd.read_csv(summary_path))
        tmp_summary = add_fix_promote_columns(tmp_summary)
        st.session_state.summary_df = tmp_summary

        st.session_state.strategy_table = None

        st.success(f"Analysis complete. Outputs saved in: {run_dir}")
        st.rerun()

    if st.session_state.get("last_run_logs") is not None:
        with st.expander("Run logs (debug)", expanded=False):
            st.write(st.session_state.last_run_logs)

    if strategy_clicked:
        if st.session_state.get("summary_df") is None or st.session_state.get("processed_df") is None:
            st.error("Run 'Sanitize + Analyze' first (outputs not loaded).")
            return
        if promo_weeks is None:
            st.error("Select number of weeks first.")
            return

        st.session_state.strategy_table = build_strategy_table(
            st.session_state.summary_df,
            st.session_state.processed_df,
            st.session_state.get("selected_businesses", []),
            promo_weeks,
        )

    analysis_ready = st.session_state.get("summary_df") is not None and st.session_state.get("processed_df") is not None
    if not analysis_ready:
        st.info("Run 'Sanitize + Analyze' first. After that, click 'Generate Strategy'.")
        return

    if st.session_state.get("strategy_table") is not None:
        st.subheader("Strategy Preview (by Theme)")
        show_table(st.session_state.strategy_table, key="strategy_preview")

        st.download_button(
            "Download strategy.csv",
            data=st.session_state.strategy_table.to_csv(index=False).encode("utf-8"),
            file_name="strategy.csv",
            mime="text/csv",
        )
    else:
        st.info("Click 'Generate Strategy' to build the strategy table and enable download.")

    with st.expander("How we derived the strategy (Themes)", expanded=False):
        show_table(st.session_state.summary_df, key="themes_table")
        st.download_button(
            "Download themes (theme_summary.csv)",
            data=st.session_state.summary_df.to_csv(index=False).encode("utf-8"),
            file_name=SUMMARY_OUT,
            mime="text/csv",
        )

    with st.expander("Sanitized dataset (Processed reviews)", expanded=False):
        show_table(st.session_state.processed_df.head(25), key="processed_preview")
        st.download_button(
            "Download processed reviews (processed_reviews.csv)",
            data=st.session_state.processed_df.to_csv(index=False).encode("utf-8"),
            file_name=PROCESSED_OUT,
            mime="text/csv",
        )


def page_dashboard():
    nav_bar("3) Dashboard")
    st.header("Quick Stats + Dashboard")

    if st.session_state.summary_df is None or st.session_state.processed_df is None:
        st.warning("Run analysis on Page 2 first.")
        return

    df = st.session_state.processed_df
    summary_df = st.session_state.summary_df

    selected = st.session_state.get("selected_businesses", [])
    selected_label = "; ".join(selected) if selected else "ALL (no filter)"
    promo_weeks = st.session_state.get("promo_weeks")
    sample_n = st.session_state.get("sample_n")
    st.caption(
        f"Dashboard reflects the **last analyzed run** for: **{selected_label}** "
        f"| Promo weeks: **{promo_weeks if promo_weeks is not None else 'Not selected'}** "
        f"| Max reviews sampled: **{sample_n}**"
    )

    st.subheader("Quick stats")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reviews analyzed", len(df))

    if "rating" in df.columns:
        avg_rating = pd.to_numeric(df["rating"], errors="coerce").mean()
        m2.metric("Avg rating", "N/A" if pd.isna(avg_rating) else round(float(avg_rating), 2))
    else:
        m2.metric("Avg rating", "N/A")

    m3.metric("Themes found", len(summary_df))
    sil = compute_silhouette_if_possible(df)
    m4.metric("Silhouette (sanity check)", "N/A" if sil is None else round(sil, 3))

    st.divider()

    st.subheader("Theme Insights (what customers talk about)")
    st.write(
        "- **Pie chart** shows the share of reviews in each theme (higher = talked about more).\n"
        "- **Bar chart** shows average rating by theme (lower = pain points to fix first)."
    )

    plot_df = summary_df.copy()
    if "Review_Count" not in plot_df.columns and "Size" in plot_df.columns:
        plot_df["Review_Count"] = plot_df["Size"]

    # Prefer simplified label on charts; fallback to Theme_Display
    if "Theme_Label_Simple" in plot_df.columns:
        plot_df["Theme_For_Chart"] = plot_df["Theme_Label_Simple"]
    elif "Theme_Display" in plot_df.columns:
        plot_df["Theme_For_Chart"] = plot_df["Theme_Display"]
    else:
        plot_df["Theme_For_Chart"] = plot_df.index.astype(str)

    c1, c2 = st.columns(2)
    with c1:
        if {"Theme_For_Chart", "Review_Count"}.issubset(plot_df.columns):
            fig_pie = px.pie(plot_df, values="Review_Count", names="Theme_For_Chart", title="Theme distribution (review volume)")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Theme distribution chart unavailable (missing counts).")

    with c2:
        if {"Theme_For_Chart", "Avg_Rating"}.issubset(plot_df.columns):
            fig_bar = px.bar(
                plot_df.sort_values("Avg_Rating"),
                x="Theme_For_Chart",
                y="Avg_Rating",
                title="Average rating by theme (lower = needs improvement)",
            )
            fig_bar.update_yaxes(range=[0, 5])
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Avg rating by theme chart unavailable (missing Avg_Rating).")

    st.divider()

    st.subheader("Additional checks")
    st.write(
        "- **Ratings histogram** shows overall customer sentiment spread.\n"
        "- **Reviews per restaurant** helps confirm your selected filter worked."
    )

    c3, c4 = st.columns(2)
    with c3:
        if "rating" in df.columns:
            tmp = df.copy()
            tmp["rating_num"] = pd.to_numeric(tmp["rating"], errors="coerce")
            tmp = tmp.dropna(subset=["rating_num"])
            if len(tmp) > 0:
                fig_hist = px.histogram(tmp, x="rating_num", nbins=10, title="Ratings distribution (overall)")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No numeric ratings available to plot.")
        else:
            st.info("Ratings histogram unavailable (missing rating column).")

    with c4:
        if "business_name" in df.columns:
            counts = df["business_name"].fillna("Unknown").value_counts().head(15).reset_index()
            counts.columns = ["business_name", "review_count"]
            fig_rest = px.bar(counts, x="business_name", y="review_count", title="Reviews per restaurant (top 15)")
            st.plotly_chart(fig_rest, use_container_width=True)
        else:
            st.info("Restaurant count chart unavailable (missing business_name column).")


# ----------------------------
# Sidebar controls + routing
# ----------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "1) Upload Reviews"

if st.session_state.get("uploaded_bytes") is None:
    st.session_state["nav_page"] = "1) Upload Reviews"

if st.sidebar.button("Reset app"):
    st.session_state.clear()
    st.rerun()

page = st.sidebar.radio("Navigate", PAGES, key="nav_page")

if page == "1) Upload Reviews":
    page_upload()
elif page == "2) Analyze & Strategy":
    page_analyze_and_strategy()
else:
    page_dashboard()