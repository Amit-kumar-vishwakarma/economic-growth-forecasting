from pathlib import Path
import re
import json, urllib.parse, urllib.request
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Economic Growth Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FEATURES = DATA / "india_economic_growth_features.csv"
FORECAST = DATA / "india_gdp_forecast.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
PAGES = ["Overview", "Economic Trends", "Country Comparison", "Model Performance", "Data & Downloads"]
INDICATORS = {
    "GDP Growth": ("GDP_Growth", "%", "Annual change in the value of goods and services produced by the economy."),
    "Inflation": ("Inflation", "%", "Annual change in consumer prices."),
    "Unemployment": ("Unemployment", "%", "Share of the labor force that is unemployed and actively seeking work."),
    "Exports (% of GDP)": ("Exports_GDP", "% of GDP", "Exports of goods and services relative to GDP."),
    "Imports (% of GDP)": ("Imports_GDP", "% of GDP", "Imports of goods and services relative to GDP."),
    "Capital Formation (% of GDP)": ("Capital_Formation_GDP", "% of GDP", "Investment in fixed assets and inventories relative to GDP."),
}
WB = {"GDP growth (%)":"NY.GDP.MKTP.KD.ZG", "Unemployment (%)":"SL.UEM.TOTL.ZS", "Inflation (%)":"FP.CPI.TOTL.ZG"}
COUNTRIES = {"India":"IND", "China":"CHN", "United States":"USA", "Japan":"JPN", "Germany":"DEU"}
METRICS = pd.DataFrame([
    {"Model":"Naive Baseline","Mean MAE":1.993,"Mean RMSE":2.642,"Mean R²":-0.804},
    {"Model":"XGBoost","Mean MAE":2.624,"Mean RMSE":3.323,"Mean R²":-5.377},
    {"Model":"Random Forest","Mean MAE":2.898,"Mean RMSE":3.507,"Mean R²":-9.863},
    {"Model":"Gradient Boosting","Mean MAE":3.687,"Mean RMSE":4.318,"Mean R²":-23.596},
    {"Model":"Linear Regression","Mean MAE":9.454,"Mean RMSE":11.269,"Mean R²":-172.139},
])
for key, value in [("theme","Dark"),("page","Overview"),("trend_indicator","GDP Growth"),("country_result",None),("country_error",None)]:
    if key not in st.session_state: st.session_state[key] = value
DARK = st.session_state.theme == "Dark"
C = {
 "bg":"#0B1410" if DARK else "#F4F6F2","surface":"#14221B" if DARK else "#FFFFFF",
 "surface2":"#1D3326" if DARK else "#EAF0E8","text":"#F4F8F4" if DARK else "#202A23",
 "muted":"#C1D0C5" if DARK else "#66756A","border":"#3A5544" if DARK else "#DCE5DA",
 "accent":"#54E58A" if DARK else "#168A45","soft":"#1B422C" if DARK else "#DDF4E4",
 "grid":"rgba(205,225,210,.22)" if DARK else "rgba(50,75,55,.13)"
}
st.markdown(f"""
<style>
:root {{color-scheme:{"dark" if DARK else "light"};}}
.stApp {{background:{C["bg"]};color:{C["text"]};background-image:radial-gradient(ellipse at 92% 0%,{"rgba(52,211,112,.09)" if DARK else "rgba(38,150,77,.07)"},transparent 34%);}}
[data-testid="stHeader"] {{background:transparent;}}
[data-testid="stSidebar"] {{background:{C["surface"]};border-right:1px solid {C["border"]};}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{font-size:.86rem;line-height:1.55;}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{color:{C["muted"]}!important;font-size:.9rem;}}
.stSelectbox label,.stMultiSelect label,.stSlider label,.stNumberInput label,.stRadio label {{color:{C["text"]}!important;font-weight:650!important;}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{color:{C["text"]}!important;}}
[data-testid="stDataFrame"] * {{color:{C["text"]}!important;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{padding:.35rem .5rem;border-radius:8px;}}
.block-container {{max-width:1600px;padding-top:.8rem;padding-bottom:2.5rem;}}
h1,h2,h3,h4 {{color:{C["text"]}!important;letter-spacing:-.035em;}}
p,label,[data-testid="stMarkdownContainer"] {{color:{C["text"]};}}
.brand {{display:flex;align-items:center;gap:.65rem;font-weight:850;letter-spacing:.08em;font-size:.78rem;color:{C["accent"]};}}
.mark {{width:34px;height:34px;display:inline-flex;align-items:center;justify-content:center;border-radius:11px;background:{C["soft"]};font-size:1.2rem;}}
.hero {{position:relative;overflow:hidden;isolation:isolate;border:1px solid {C["border"]};border-radius:18px;padding:1.2rem 1.45rem;margin:.45rem 0 .8rem;background:linear-gradient(112deg,{C["surface"]},{C["surface2"]});min-height:145px;}}
.hero:after {{content:"";position:absolute;z-index:-1;right:2%;top:0;width:360px;height:100%;opacity:{"0.23" if DARK else "0.14"};background-repeat:no-repeat;background-position:center right;background-size:contain;background-image:url("data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 600 300%27 fill=%27%2343d477%27 fill-opacity=%270.22%27 stroke=%27%2343d477%27 stroke-width=%273%27 stroke-linejoin=%27round%27%3E%3Cpath d=%27M45 69l32-24 48 7 20 22 35 7 10 24-18 17-9 29-28 9-8 33-22 16-15-25-2-33-22-22-6-29-25-10z M191 167l29-16 24 13 8 31-15 27-7 34-19 18-14-35-13-21z M267 54l36-20 43 8 22-12 43 15 27-5 25 20 37 3 27 22-17 21-35 2-17 17-30-5-17 19-22-9-17 24-19-4-12-28-23-5-12-27-28-8-12-21-25-1z M315 159l30-12 25 18 19 28-11 35-25 34-20-19-8-33-20-22z M460 210l24-12 26 13 10 22-18 20-26-7z M522 113l20-10 20 16-8 20-20-3z%27/%3E%3C/svg%3E");pointer-events:none;}}
.eyebrow,.kicker {{color:{C["accent"]};text-transform:uppercase;letter-spacing:.15em;font-size:.68rem;font-weight:850;}}
.hero-title {{font-size:clamp(1.8rem,3vw,2.6rem);line-height:1.08;font-weight:850;margin:.45rem 0;max-width:750px;}}
.hero-sub {{color:{C["muted"]};max-width:660px;font-size:1rem;line-height:1.65;font-weight:500;}}
.kpi-label {{color:{C["muted"]};font-size:.74rem;font-weight:850;letter-spacing:.075em;text-transform:uppercase;}}
.kpi-note {{color:{C["accent"]};font-size:.82rem;font-weight:650;}}
[data-testid="stDataFrame"] {{background:{C["surface"]};}}
[data-testid="stDataFrame"] [role="columnheader"] {{background:{C["surface2"]}!important;}}
.stMarkdown, .stMarkdown p, .stMarkdown li {{line-height:1.55;}}
.stButton>button, .stDownloadButton>button {{color:{C["text"]};background:{C["surface2"]};}}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{background:{C["surface"]};border-color:{C["border"]};}}
[data-baseweb="select"] *, [data-baseweb="input"] input {{color:{C["text"]}!important;}}
/* Forecast year picker: one editable year field and exactly two explicit arrow buttons. */
[data-testid="stTextInput"] label {{color:{C["text"]}!important;font-size:1rem!important;font-weight:750!important;}}
[data-testid="stTextInput"] [data-baseweb="input"] {{background:{C["surface"]}!important;border:1.5px solid {C["border"]}!important;border-radius:12px!important;min-height:60px!important;box-shadow:none!important;}}
[data-testid="stTextInput"] input {{background:{C["surface"]}!important;background-color:{C["surface"]}!important;color:{C["text"]}!important;-webkit-text-fill-color:{C["text"]}!important;caret-color:{C["accent"]}!important;font-size:1.7rem!important;font-weight:850!important;min-height:56px!important;opacity:1!important;}}
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {{border-color:{C["accent"]}!important;box-shadow:0 0 0 2px {C["soft"]}!important;}}
[class*="st-key-year-step-down"] button, [class*="st-key-year-step-up"] button {{height:60px!important;min-height:60px!important;width:100%!important;font-size:1.45rem!important;font-weight:900!important;background:{C["surface2"]}!important;color:{C["text"]}!important;border:1px solid {C["border"]}!important;border-radius:11px!important;padding:0!important;}}
[class*="st-key-year-step-down"] button:hover, [class*="st-key-year-step-up"] button:hover {{background:{C["accent"]}!important;color:#FFFFFF!important;border-color:{C["accent"]}!important;}}



.kpi {{border:1px solid {C["border"]};border-radius:13px;background:{C["surface"]};padding:.85rem;min-height:108px;transition:transform .18s,border-color .18s;}}
.kpi:hover {{transform:translateY(-2px);border-color:{C["accent"]};}}
.kpi-label {{color:{C["muted"]};font-size:.68rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}}
.kpi-value {{color:{C["text"]};font-size:1.65rem;font-weight:850;letter-spacing:-.04em;margin:.35rem 0;overflow-wrap:anywhere;}}
.kpi-note {{color:{C["accent"]};font-size:.76rem;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{border-color:{C["border"]}!important;border-radius:16px!important;background:{C["surface"]};}}
div[data-testid="stMetric"] {{background:{C["surface"]};border:1px solid {C["border"]};border-radius:13px;padding:.8rem;}}
div[data-testid="stMetricLabel"] {{color:{C["muted"]}!important;}} div[data-testid="stMetricValue"] {{color:{C["text"]}!important;}}
[data-testid="stDataFrame"] {{border:1px solid {C["border"]};border-radius:10px;overflow:hidden;}}
.stButton>button,.stDownloadButton>button {{border-radius:10px;font-weight:700;border:1px solid {C["border"]};min-height:40px;}}
.stButton>button:hover,.stDownloadButton>button:hover {{border-color:{C["accent"]};color:{C["accent"]};}}
hr {{border-color:{C["border"]};}}
/* Dark-mode readability: target Streamlit's nested widgets as well as page text. */
[data-testid="stAppViewContainer"] {{color:{C["text"]}!important;}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
[data-testid="stToggle"] label {{color:{C["text"]}!important;}}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *,
small {{color:{C["muted"]}!important;}}
[data-testid="stTabs"] button {{color:{C["text"]}!important;}}
[data-testid="stExpander"] {{border-color:{C["border"]}!important;}}
[data-testid="stAlert"] {{background:{C["surface"]};}}
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="columnheader"] {{color:{C["text"]}!important;}}
/* Light theme only: force readable contrast without changing the existing layout/theme design. */
{'' if DARK else r"""
[data-testid="stSidebar"], [data-testid="stSidebar"] * { color:#202A23 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#526257 !important; opacity:1 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] label { color:#202A23 !important; }
[data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li, [data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span, [data-testid="stAppViewContainer"] strong,
[data-testid="stAppViewContainer"] em { color:#202A23; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
.stCaption, small, .hero-sub, .section-caption { color:#526257 !important; opacity:1 !important; }
[data-testid="stDataFrame"], [data-testid="stDataFrame"] *,
[data-testid="stTable"] th, [data-testid="stTable"] td { color:#202A23 !important; }
[data-testid="stDataFrame"] [role="columnheader"] { background:#EAF0E8 !important; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="input"] > div,
[data-baseweb="textarea"] > div { background:#FFFFFF !important; background-color:#FFFFFF !important; border-color:#C8D3CA !important; }
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="input"] > div,
[data-testid="stTextInput"] input {
  background:#FFFFFF !important; background-color:#FFFFFF !important;
  color:#202A23 !important; -webkit-text-fill-color:#202A23 !important;
  opacity:1 !important; box-shadow:none !important;
}
[data-baseweb="select"] *, [data-baseweb="input"] input,
[data-baseweb="textarea"] textarea { color:#202A23 !important; -webkit-text-fill-color:#202A23 !important; }
[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"],
li[role="option"], [role="option"] { background:#FFFFFF !important; color:#202A23 !important; }
.stButton > button, .stDownloadButton > button { background:#FFFFFF !important; color:#202A23 !important; border-color:#C8D3CA !important; }
.stButton > button:hover, .stDownloadButton > button:hover { color:#126B3A !important; border-color:#168A45 !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color:#202A23 !important; }
[data-testid="stDataFrame"] { background:#FFFFFF !important; border:1px solid #DCE5DA !important; }
[data-testid="stDataFrame"] [role="grid"], [data-testid="stDataFrame"] [role="row"],
[data-testid="stDataFrame"] [role="gridcell"] {
  background:#FFFFFF !important; color:#202A23 !important;
  border-bottom:1px solid #B8C7BB !important;
  border-right:1px solid #B8C7BB !important;
  box-shadow:inset 0 -1px 0 #B8C7BB, inset -1px 0 0 #B8C7BB !important;
}
[data-testid="stDataFrame"] [role="columnheader"] {
  background:#EAF0E8 !important; color:#202A23 !important;
  border-bottom:1px solid #A9BAAD !important;
  border-right:1px solid #B8C7BB !important;
  box-shadow:inset 0 -1px 0 #A9BAAD, inset -1px 0 0 #B8C7BB !important;
}
[data-testid="stDataFrame"] [role="row"] { border-bottom:1px solid #B8C7BB !important; }
[data-testid="stTable"] table, [data-testid="stTable"] th, [data-testid="stTable"] td {
  background:#FFFFFF !important; color:#202A23 !important; border-color:#DCE5DA !important;
}
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] {
  background:#EAF0E8 !important; border:1px solid #C8D3CA !important; border-radius:11px !important; padding:3px !important;
}
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button {
  min-height:36px !important; border-radius:8px !important; font-weight:750 !important;
  color:#202A23 !important; background:#EAF0E8 !important;
}
[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-checked="true"] {
  background:#168A45 !important; color:#FFFFFF !important; border:1px solid #168A45 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#39483D !important; opacity:1 !important; }
"""}
@media(max-width:800px) {{.block-container {{padding-left:1rem;padding-right:1rem;}}.hero {{padding:1.15rem;}}.hero:after {{right:-150px;opacity:.5;}}}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_csv(path): return pd.read_csv(path)
def load(path, label):
    if not path.exists(): return None, f"{label} not found: {path.relative_to(ROOT)}"
    try:
        df=read_csv(str(path))
        return df, (f"{label} is empty." if df.empty else None)
    except Exception as exc: return None, f"Could not read {label}: {exc}"
features, fe = load(FEATURES, "Processed dataset")
forecast, fo = load(FORECAST, "Forecast file")
if features is not None and "Year" in features:
    features["Year"]=pd.to_numeric(features["Year"],errors="coerce")
    features=features.dropna(subset=["Year"]).copy()
    features["Year"]=features["Year"].astype(int)
    features=features.sort_values("Year").reset_index(drop=True)

def num(row, col):
    if row is None or col not in row.index: return None
    v=pd.to_numeric(pd.Series([row[col]]),errors="coerce").iloc[0]
    return None if pd.isna(v) else float(v)
def fmt(v, suffix=""): return "Data unavailable" if v is None or pd.isna(v) else f"{v:,.2f}{suffix}"
def latest_row(df,col):
    if df is None or df.empty or "Year" not in df or col not in df: return None
    d=df.copy(); d[col]=pd.to_numeric(d[col],errors="coerce"); d=d.dropna(subset=[col])
    return None if d.empty else d.iloc[-1]
def available(df): return {k:v for k,v in INDICATORS.items() if df is not None and v[0] in df.columns}


def show_table(df):
    """Render every table through the same Styler-backed HTML table in both themes."""
    if df is None:
        st.info("No data available.")
        return
    if isinstance(df, pd.DataFrame):
        display_df = df.copy()
    elif hasattr(df, "data") and isinstance(getattr(df, "data", None), pd.DataFrame):
        display_df = df.data.copy()
    else:
        display_df = pd.DataFrame(df)

    if display_df.empty:
        st.info("No rows to display.")
        return

    # Use one renderer in Light and Dark modes. This avoids the previous
    # mismatch where Light used st.table and Dark used st.dataframe.
    header_bg = C["surface2"]
    body_bg = C["surface"]
    text_color = C["text"]
    border_color = "#A9BAAD" if not DARK else "#3A5544"
    header_border = "#8FA493" if not DARK else "#4B6A55"
    alternate_bg = "#F5F8F4" if not DARK else "#192B21"

    styled_df = display_df.style.set_properties(
        **{
            "background-color": body_bg,
            "color": text_color,
            "border": f"1px solid {border_color}",
            "padding": "0.45rem 0.6rem",
            "width": "33.3333%",
            "overflow-wrap": "anywhere",
        }
    ).set_table_styles([
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("table-layout", "fixed"),
            ("width", "100%"),
            ("border", f"1px solid {border_color}"),
        ]},
        {"selector": "thead th", "props": [
            ("background-color", header_bg),
            ("color", text_color),
            ("border", f"1px solid {header_border}"),
            ("padding", "0.55rem 0.6rem"),
            ("text-align", "left"),
            ("width", f"{100 / max(1, len(display_df.columns)):.4f}%"),
            ("overflow-wrap", "anywhere"),
        ]},
        {"selector": "tbody td", "props": [
            ("background-color", body_bg),
            ("color", text_color),
            ("border", f"1px solid {border_color}"),
            ("padding", "0.45rem 0.6rem"),
            ("width", f"{100 / max(1, len(display_df.columns)):.4f}%"),
            ("overflow-wrap", "anywhere"),
        ]},
        {"selector": "tbody tr:nth-child(even) td", "props": [
            ("background-color", alternate_bg),
        ]},
    ])
    st.table(styled_df)

def _normal_name(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())

def _lag_spec(value):
    """Return (normalized base feature, lag number) for names such as GDP_Growth_Lag1."""
    name = _normal_name(value)
    for lag in (1, 2):
        suffix = f"lag{lag}"
        if name.endswith(suffix):
            return name[:-len(suffix)], lag
    return None, None

def _find_raw_column(base_normalized, columns):
    for candidate in columns:
        candidate_base, candidate_lag = _lag_spec(candidate)
        if candidate_lag is None and _normal_name(candidate) == base_normalized:
            return candidate
    return None

def _trend_value(column, target_year, window=10):
    """Damped, recent-trend estimate used for non-GDP indicators and transparent fallback scenarios."""
    if features is None or features.empty or column not in features.columns or "Year" not in features.columns:
        return None
    series = features[["Year", column]].copy()
    series["Year"] = pd.to_numeric(series["Year"], errors="coerce")
    series[column] = pd.to_numeric(series[column], errors="coerce")
    series = series.dropna().sort_values("Year").tail(window)
    if series.empty:
        return None
    last_year = int(series["Year"].iloc[-1])
    last_value = float(series[column].iloc[-1])
    horizon = int(target_year) - last_year
    if horizon <= 0:
        row = features.loc[features["Year"] == int(target_year), column]
        return float(row.dropna().iloc[0]) if not row.dropna().empty else None
    if len(series) < 2:
        return last_value
    x = series["Year"].to_numpy(dtype=float)
    y = series[column].to_numpy(dtype=float)
    # Combine a 10-year trend with the more recent 5-year trend to avoid overreacting to one point.
    long_slope = float(np.polyfit(x, y, 1)[0])
    recent = series.tail(min(5, len(series)))
    recent_slope = float(np.polyfit(recent["Year"].to_numpy(dtype=float), recent[column].to_numpy(dtype=float), 1)[0]) if len(recent) >= 2 else long_slope
    slope = 0.60 * recent_slope + 0.40 * long_slope
    # Damping ensures the line keeps changing by year but does not grow without bound.
    change = slope * sum(0.78 ** step for step in range(horizon))
    recent_std = float(series[column].std(ddof=0)) if len(series) > 1 else 0.0
    limit = max(1.0, recent_std * 2.25)
    change = float(np.clip(change, -limit, limit))
    estimate = last_value + change
    # If the fitted trend is perfectly flat, use a small, explicitly illustrative drift rather than repeating the same value.
    if abs(estimate - last_value) < 1e-6:
        recent_delta = float(y[-1] - y[-2]) if len(y) > 1 else 0.0
        direction = np.sign(recent_delta) if abs(recent_delta) > 1e-6 else 1.0
        estimate = last_value + direction * min(0.08 * horizon, 0.4)
    return float(estimate)


def _illustrative_gdp_path(target_year, reason="The trained model could not produce a usable forecast path"):
    """Clearly-labelled trend-scenario fallback; not presented as a validated ML forecast."""
    if features is None or features.empty or "GDP_Growth" not in features.columns:
        return pd.DataFrame(columns=["Year", "Predicted GDP growth (%)"]), reason + "; GDP history is unavailable."
    last_year = int(features["Year"].max())
    rows = []
    for year in range(last_year + 1, int(target_year) + 1):
        value = _trend_value("GDP_Growth", year)
        if value is not None and np.isfinite(value):
            rows.append({"Year": year, "Predicted GDP growth (%)": value})
    if not rows:
        return pd.DataFrame(columns=["Year", "Predicted GDP growth (%)"]), reason + "; no fallback estimate could be calculated."
    return pd.DataFrame(rows), "Illustrative trend-scenario estimates (fallback; not a validated model forecast)"


def _model_forecast_path(target_year):
    """Recursive saved-model path. If the model fails or is flat across future years, clearly switch to a trend scenario."""
    empty = pd.DataFrame(columns=["Year", "Predicted GDP growth (%)"])
    if features is None or features.empty or "Year" not in features.columns or "GDP_Growth" not in features.columns:
        return empty, "Processed historical GDP features are unavailable."
    last_year = int(features["Year"].max())
    if int(target_year) <= last_year:
        return empty, "Selected year is historical; actual observations are shown instead of a forecast."
    if not MODEL_PATH.exists():
        return _illustrative_gdp_path(target_year, "Saved model file was not found")
    try:
        model = joblib.load(MODEL_PATH)
        trained_names = getattr(model, "feature_names_in_", None)
        model_features = list(trained_names) if trained_names is not None else []
        if not model_features and hasattr(model, "get_booster"):
            booster_names = model.get_booster().feature_names
            model_features = list(booster_names) if booster_names else []
        if not model_features:
            return _illustrative_gdp_path(target_year, "Model feature names are unavailable")

        history = features.sort_values("Year").reset_index(drop=True).copy()
        predictions = []
        for year in range(last_year + 1, int(target_year) + 1):
            latest = history.iloc[-1]
            inputs = {}
            for name in model_features:
                base, lag = _lag_spec(name)
                if lag is not None:
                    raw_col = _find_raw_column(base, history.columns)
                    if raw_col is None or len(history) < lag:
                        return _illustrative_gdp_path(target_year, f"Lag feature {name} could not be rebuilt")
                    inputs[name] = pd.to_numeric(history.iloc[-lag][raw_col], errors="coerce")
                elif _normal_name(name) == "year":
                    inputs[name] = year
                else:
                    match = name if name in history.columns else next((c for c in history.columns if _normal_name(c) == _normal_name(name)), None)
                    if match is None:
                        return _illustrative_gdp_path(target_year, f"Model feature {name} is unavailable")
                    inputs[name] = latest[match]
            X = pd.DataFrame([[inputs[name] for name in model_features]], columns=model_features).apply(pd.to_numeric, errors="coerce")
            if X.isna().any().any():
                return _illustrative_gdp_path(target_year, "Required model inputs contain missing values")
            prediction = float(model.predict(X)[0])
            if not np.isfinite(prediction):
                return _illustrative_gdp_path(target_year, "Model returned a non-finite value")
            predictions.append({"Year": year, "Predicted GDP growth (%)": prediction})

            # Keep macro indicators fixed at their latest actual value, but roll GDP lags forward recursively.
            new_row = latest.copy()
            new_row["Year"] = year
            new_row["GDP_Growth"] = prediction
            for feature_col in history.columns:
                base, lag = _lag_spec(feature_col)
                raw_col = _find_raw_column(base, history.columns) if lag is not None else None
                if raw_col is not None and len(history) >= lag:
                    new_row[feature_col] = history.iloc[-lag][raw_col]
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        path = pd.DataFrame(predictions, columns=["Year", "Predicted GDP growth (%)"])
        # A flat tree-model path looks dynamic in the UI but does not respond to year. Blend with a damped trend estimate.
        if len(path) >= 1 and float(path["Predicted GDP growth (%)"].max() - path["Predicted GDP growth (%)"].min()) < 0.08:
            blended = []
            for year_value, model_value in zip(path["Year"], path["Predicted GDP growth (%)"]):
                trend = _trend_value("GDP_Growth", int(year_value))
                base = float(model_value)
                blended.append(0.55 * base + 0.45 * float(trend if trend is not None else base))
            path["Predicted GDP growth (%)"] = blended
            return path, "Illustrative trend-adjusted model estimates (saved model was nearly flat; not a validated multi-year forecast)"
        return path, "Recursive saved-model estimates; non-GDP inputs held at latest observed values"
    except Exception as exc:
        return _illustrative_gdp_path(target_year, f"Saved model forecast failed ({type(exc).__name__})")


def forecast_for_year(target_year):
    """Return historical actual for historical years, model/trend estimate for future years."""
    if features is None or features.empty:
        return None, "Processed historical features are unavailable."
    year = int(target_year)
    last_year = int(features["Year"].max())
    if year <= last_year:
        rows = features.loc[features["Year"] == year]
        if rows.empty:
            return None, f"No historical observation is available for {year}."
        value = num(rows.iloc[0], "GDP_Growth")
        return value, "Historical actual" if value is not None else f"GDP growth is missing for {year}."
    path, status = _model_forecast_path(year)
    row = path.loc[path["Year"] == year] if not path.empty else path
    if row.empty:
        return None, status
    return float(row.iloc[0]["Predicted GDP growth (%)"]), status


def forecast_path(target_year):
    if features is None or features.empty:
        return pd.DataFrame(columns=["Year", "Predicted GDP growth (%)"]), "Historical dataset unavailable."
    if int(target_year) <= int(features["Year"].max()):
        return pd.DataFrame(columns=["Year", "Predicted GDP growth (%)"]), "Selected year is historical; showing actual observations."
    path, status = _model_forecast_path(int(target_year))
    if path.empty or int(path["Year"].max()) != int(target_year):
        return _illustrative_gdp_path(int(target_year), status)
    return path, status

def data_for(df,label,col,start=None,end=None):
    if df is None or df.empty or "Year" not in df or col not in df: return pd.DataFrame(columns=["Year",label])
    d=df[["Year",col]].copy(); d[col]=pd.to_numeric(d[col],errors="coerce"); d=d.dropna(subset=[col])
    if start is not None: d=d[d.Year>=start]
    if end is not None: d=d[d.Year<=end]
    return d.rename(columns={col:label}).sort_values("Year")
def style(fig,height=390):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family":"Inter,Arial,sans-serif","color":C["text"],"size":14},
        margin={"l":20,"r":20,"t":54,"b":32}, height=height,
        hovermode="x unified", transition={"duration":250},
        hoverlabel={"bgcolor":C["surface"],"font_color":C["text"],"bordercolor":C["border"],"font_size":14},
        xaxis={"showgrid":False,"zeroline":False,"linecolor":C["border"],
               "tickfont":{"color":C["text"],"size":13},
               "title_font":{"color":C["text"],"size":14}},
        yaxis={"gridcolor":C["grid"],"zeroline":False,
               "tickfont":{"color":C["text"],"size":13},
               "title_font":{"color":C["text"],"size":14}},
        legend={"orientation":"h","yanchor":"bottom","y":1.02,"xanchor":"right","x":1,
                "font":{"color":C["text"],"size":13}}
    )
    return fig
@st.cache_data(ttl=86400,show_spinner=False)
def world_bank(indicator,codes,start,end):
    url="https://api.worldbank.org/v2/country/"+urllib.parse.quote(";".join(codes),safe=";")+"/indicator/"+urllib.parse.quote(indicator)+"?"+urllib.parse.urlencode({"date":f"{start}:{end}","format":"json","per_page":2000})
    req=urllib.request.Request(url,headers={"User-Agent":"EconomicGrowthIntelligence/1.0"})
    with urllib.request.urlopen(req,timeout=20) as resp: payload=json.loads(resp.read().decode("utf-8"))
    rows=[]
    if isinstance(payload,list) and len(payload)>1 and isinstance(payload[1],list):
        for x in payload[1]:
            if x.get("value") is not None: rows.append({"Country":x.get("country",{}).get("value","Unknown"),"Year":int(x["date"]),"Value":float(x["value"])})
    return pd.DataFrame(rows,columns=["Country","Year","Value"])

# App shell: compact left navigation + independent theme switch
with st.sidebar:
    st.markdown('<div class="brand"><span class="mark">◈</span> ECONOMIC<br> GROWTH INTELLIGENCE</div>',unsafe_allow_html=True)
    st.caption("MACROECONOMIC ANALYTICS")
    st.divider()
    st.radio("WORKSPACE",PAGES,key="page",label_visibility="collapsed")
    st.divider()
    st.markdown("**APPEARANCE**")
    theme=st.segmented_control("Theme",["Dark","Light"],default=st.session_state.theme,key="theme_picker",label_visibility="collapsed")
    if theme and theme!=st.session_state.theme:
        st.session_state.theme=theme
        st.rerun()
    st.markdown("---")
    st.markdown(f'<div class="kicker">DATA SOURCES</div><p style="font-size:.82rem;color:{C["muted"]};opacity:1">India indicators · saved project CSV<br>Country comparison · World Bank API<br>Forecast · saved model output</p>',unsafe_allow_html=True)
st.markdown(f"""<style>
/* Keep the year controls readable in both themes; scoped only to the two year-step buttons. */
[data-testid="stRadio"] label, [data-testid="stSegmentedControl"] label,
[data-testid="stWidgetLabel"] p {{ opacity: 1 !important; }}
[data-testid="stSegmentedControl"] button {{ font-weight: 700 !important; }}
[class*="st-key-year-step-down"] [data-testid="stButton"] > button,
[class*="st-key-year-step-up"] [data-testid="stButton"] > button {{
  background: {C["surface2"]} !important;
  background-color: {C["surface2"]} !important;
  color: {C["text"]} !important;
  -webkit-text-fill-color: {C["text"]} !important;
  border: 2px solid {C["border"]} !important;
  opacity: 1 !important;
  box-shadow: none !important;
  filter: none !important;
}}
[class*="st-key-year-step-down"] [data-testid="stButton"] > button *,
[class*="st-key-year-step-up"] [data-testid="stButton"] > button * {{
  color: {C["text"]} !important;
  -webkit-text-fill-color: {C["text"]} !important;
  fill: {C["text"]} !important;
  opacity: 1 !important;
}}
[class*="st-key-year-step-down"] [data-testid="stButton"] > button:hover,
[class*="st-key-year-step-up"] [data-testid="stButton"] > button:hover {{
  background: {C["accent"]} !important;
  background-color: {C["accent"]} !important;
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
  border-color: {C["accent"]} !important;
}}
[class*="st-key-year-step-down"] [data-testid="stButton"] > button:hover *,
[class*="st-key-year-step-up"] [data-testid="stButton"] > button:hover * {{
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
  fill: #FFFFFF !important;
}}
</style>""", unsafe_allow_html=True)
page=st.session_state.page
st.markdown('<div class="brand"><span class="mark">◈</span> ECONOMIC GROWTH INTELLIGENCE <span style="margin-left:auto;color:'+C["muted"]+';font-size:.75rem;font-weight:500;letter-spacing:0">MACROECONOMIC WORKSPACE</span></div>',unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="eyebrow">🇮🇳 INDIA · MACROECONOMIC OUTLOOK</div><div class="hero-title">Understand the forces shaping growth.</div><div class="hero-sub">Explore India’s economic indicators, compare global economies, and review a data-driven GDP growth estimate.</div></div>',unsafe_allow_html=True)
if fe or fo:
    with st.expander("Data availability",expanded=features is None or forecast is None):
        if fe: st.warning(fe)
        if fo: st.warning(fo)

gdp_row=latest_row(features,"GDP_Growth"); unemp_row=latest_row(features,"Unemployment"); inf_row=latest_row(features,"Inflation")
gdp=num(gdp_row,"GDP_Growth"); unemp=num(unemp_row,"Unemployment"); infl=num(inf_row,"Inflation")
gdp_year=int(gdp_row.Year) if gdp_row is not None else None
unemp_year=int(unemp_row.Year) if unemp_row is not None else None
infl_year=int(inf_row.Year) if inf_row is not None else None
fr=forecast.iloc[0] if forecast is not None and not forecast.empty else None
fv=num(fr,"Predicted_GDP_Growth_Percent"); fyv=num(fr,"Forecast_Year"); fy=int(fyv) if fyv is not None else None

if page=="Overview":
    st.markdown('<div class="kicker">THE BIG PICTURE</div>', unsafe_allow_html=True)
    if features is None or features.empty or "Year" not in features.columns:
        st.error("Historical dataset with a Year column is unavailable; the forecast selector cannot be initialized.")
    else:
        first_data_year = int(features["Year"].min())
        latest_data_year = int(features["Year"].max())
        default_target_year = max(latest_data_year + 1, fy or latest_data_year + 1)
        default_target_year = min(default_target_year, latest_data_year + 10)

        year_col, note_col = st.columns([1, 2], gap="medium")
        with year_col:
            st.session_state.setdefault("forecast_year_text", str(default_target_year))
            st.session_state.setdefault("last_valid_forecast_year", default_target_year)

            def _step_forecast_year(delta, lower_bound, upper_bound, default_year):
                current_raw = st.session_state.get("forecast_year_text", str(default_year))
                try:
                    current_year = int(str(current_raw).strip())
                except (TypeError, ValueError):
                    current_year = int(st.session_state.get("last_valid_forecast_year", default_year))
                new_year = max(lower_bound, min(upper_bound, current_year + delta))
                st.session_state["forecast_year_text"] = str(new_year)
                st.session_state["last_valid_forecast_year"] = new_year

            input_col, down_col, up_col = st.columns([5.2, 0.72, 0.72], gap="small", vertical_alignment="bottom")
            with input_col:
                year_text = st.text_input(
                    "Year to inspect / forecast",
                    key="forecast_year_text",
                    help=f"Enter any year from {first_data_year} through {latest_data_year + 10}. Historical years show actual observations; future years show model or labelled scenario estimates."
                )
            with down_col:
                with st.container(key="year-step-down"):
                    st.button("↓", key="forecast_year_down", use_container_width=True,
                              on_click=_step_forecast_year, args=(-1, first_data_year, latest_data_year + 10, default_target_year))
            with up_col:
                with st.container(key="year-step-up"):
                    st.button("↑", key="forecast_year_up", use_container_width=True,
                              on_click=_step_forecast_year, args=(1, first_data_year, latest_data_year + 10, default_target_year))

            try:
                selected_year = int(str(year_text).strip())
                if selected_year < first_data_year or selected_year > latest_data_year + 10:
                    raise ValueError
                st.session_state["last_valid_forecast_year"] = selected_year
            except (TypeError, ValueError):
                selected_year = int(st.session_state.get("last_valid_forecast_year", default_target_year))
                st.warning(f"Enter a whole year from {first_data_year} to {latest_data_year + 10}.")
        is_historical = selected_year <= latest_data_year

        full_history = data_for(features, "GDP growth (%)", "GDP_Growth")
        future_path = pd.DataFrame(columns=["Year", "Predicted GDP growth (%)"])
        if is_historical:
            selected_rows = features.loc[features["Year"] == selected_year]
            selected_gdp = num(selected_rows.iloc[0], "GDP_Growth") if not selected_rows.empty else None
            selected_status = "Historical actual" if selected_gdp is not None else f"No GDP growth observation is available for {selected_year}."
            displayed_history = full_history.loc[full_history["Year"] <= selected_year].copy()
        else:
            future_path, path_status = forecast_path(selected_year)
            if not future_path.empty:
                value_row = future_path.loc[future_path["Year"] == selected_year]
                selected_gdp = float(value_row.iloc[0]["Predicted GDP growth (%)"]) if not value_row.empty else None
                selected_status = path_status if selected_gdp is not None else path_status
            else:
                selected_gdp = None
                selected_status = path_status
            displayed_history = full_history.copy()

        with note_col:
            if is_historical:
                st.caption(f"{selected_year} is in the historical dataset. The dashboard shows the observed GDP growth for that year, not a forecast.")
            else:
                st.caption("For future years, GDP uses the saved model when it responds to the year; a clearly labelled trend-scenario fallback is used if the model is unavailable or nearly flat. Other indicators use damped historical-trend illustrations, not validated forecasts.")
            if selected_gdp is None:
                st.warning(selected_status)
            else:
                value_label = "Historical GDP growth" if is_historical else "GDP growth estimate"
                st.caption(f"{value_label} for {selected_year}: {selected_gdp:.2f}%.")

        def selected_indicator_value(column):
            if is_historical:
                match = features.loc[features["Year"] == selected_year]
                if match.empty:
                    return None, f"Unavailable in {selected_year}"
                value = num(match.iloc[0], column)
                return value, f"Observed in {selected_year}" if value is not None else f"Unavailable in {selected_year}"
            value = _trend_value(column, selected_year)
            return value, f"Trend-based illustration · {selected_year}" if value is not None else "Projection unavailable"

        shown_infl, shown_infl_note = selected_indicator_value("Inflation")
        shown_unemp, shown_unemp_note = selected_indicator_value("Unemployment")
        shown_capital, shown_capital_note = selected_indicator_value("Capital_Formation_GDP")
        if not is_historical and selected_status.startswith("Illustrative"):
            st.info("The saved model was unavailable or nearly flat, so this view is using a trend-scenario estimate to remain interactive. It is illustrative, not a validated model forecast.")

        selected_label = "GDP growth · selected year" if is_historical else "GDP growth forecast"
        selected_note = f"{selected_year} · historical actual" if is_historical else f"{selected_year} · {selected_status.lower()}"
        cards = [
            (selected_label, fmt(selected_gdp, "%"), selected_note if selected_gdp is not None else "Value unavailable"),
            ("Inflation", fmt(shown_infl, "%"), shown_infl_note),
            ("Unemployment", fmt(shown_unemp, "%"), shown_unemp_note),
            ("Capital formation", fmt(shown_capital, "% of GDP"), shown_capital_note),
        ]
        cols = st.columns(4, gap="small")
        for i, (label, value, note) in enumerate(cards):
            with cols[i]:
                st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)
                if i == 0:
                    indicator = "GDP Growth"
                elif i == 1:
                    indicator = "Inflation"
                elif i == 2:
                    indicator = "Unemployment"
                else:
                    indicator = "Capital Formation (% of GDP)"
                st.button("Explore indicator ↗", key=f"kpi_{i}", use_container_width=True,
                          on_click=lambda p="Economic Trends", ind=indicator: st.session_state.update({"page": p, "trend_indicator": ind}))
        st.caption("Historical values are recorded observations. Future values may include illustrative trend-scenario estimates and are not official economic projections.")
        st.divider()

        # Full-width chart uses the available space instead of sharing a narrow 2:1 card.
        with st.container(border=True):
            st.markdown("#### GDP growth trajectory")
            if is_historical:
                st.caption(f"Historical GDP growth observations through {selected_year}")
            else:
                st.caption(f"Historical observations through {latest_data_year} plus the selected-year path to {selected_year}")
            if not displayed_history.empty:
                fig = px.line(displayed_history, x="Year", y="GDP growth (%)", markers=True)
                fig.update_traces(name="Historical GDP growth", line={"color": C["accent"], "width": 3}, marker={"size": 5})
                if not is_historical and not future_path.empty:
                    pred = future_path.rename(columns={"Predicted GDP growth (%)": "GDP growth (%)"}).copy()
                    anchor = full_history.tail(1).copy()
                    # Join the last observed point to the future path so the forecast line is visibly connected.
                    pred_line = pd.concat([anchor[["Year", "GDP growth (%)"]], pred[["Year", "GDP growth (%)"]]], ignore_index=True)
                    fig.add_scatter(
                        x=pred_line["Year"], y=pred_line["GDP growth (%)"], mode="lines+markers",
                        name="Forecast / scenario", line={"color": "#D97706" if not DARK else "#FBBF24", "width": 4, "dash": "dash"},
                        marker={"size": 8},
                    )
                    fig.add_vline(x=int(full_history["Year"].max()) + 0.5, line_dash="dot",
                                  line_color="#D97706" if not DARK else "#FBBF24", opacity=0.8)
                fig.update_yaxes(title="GDP growth (%)")
                st.plotly_chart(style(fig, 455), use_container_width=True, key="overview_gdp")
            else:
                st.info(f"No historical GDP growth observations are available through {selected_year}.")

        # Keep the entire trajectory table full-width and immediately below its chart.
        historical_table = displayed_history.copy()
        historical_table["Record type"] = "Historical actual"
        if not is_historical and not future_path.empty:
            predicted_table = future_path.rename(columns={"Predicted GDP growth (%)": "GDP growth (%)"}).copy()
            predicted_table["Record type"] = path_status
            trajectory = pd.concat([historical_table, predicted_table], ignore_index=True)
        else:
            trajectory = historical_table
        if not trajectory.empty:
            st.markdown("#### GDP growth history and forecast" if not is_historical else f"#### GDP growth through {selected_year}")
            show_table(trajectory)
            st.download_button(
                "Download selected-year trajectory CSV",
                trajectory.to_csv(index=False).encode("utf-8"),
                f"gdp_growth_through_{selected_year}.csv", "text/csv", key="overview_trajectory_csv"
            )

        # Snapshot values change together with the year selector for every available indicator.
        snapshot_rows = [{"Indicator": "GDP growth (%)", "Year": selected_year,
                          "Value": selected_gdp, "Basis": "Historical actual" if is_historical else selected_status}]
        for label, column, unit in [
            ("Inflation (%)", "Inflation", "%"),
            ("Unemployment (%)", "Unemployment", "%"),
            ("Exports (% of GDP)", "Exports_GDP", "% of GDP"),
            ("Imports (% of GDP)", "Imports_GDP", "% of GDP"),
            ("Capital formation (% of GDP)", "Capital_Formation_GDP", "% of GDP"),
        ]:
            value, basis = selected_indicator_value(column)
            snapshot_rows.append({"Indicator": label, "Year": selected_year, "Value": value, "Basis": basis})
        with st.container(border=True):
            st.markdown("#### Selected-year economic snapshot")
            st.caption("Historical years show recorded values. Future non-GDP figures are trend-based illustrations, not trained-model forecasts.")
            show_table(pd.DataFrame(snapshot_rows))
            st.download_button("Download selected-year snapshot CSV", pd.DataFrame(snapshot_rows).to_csv(index=False).encode("utf-8"),
                               f"economic_snapshot_{selected_year}.csv", "text/csv", key="selected_snapshot_csv")

        with st.container(border=True):
            st.markdown("#### Indicator pulse")
            opts = available(features)
            if opts:
                selected = st.selectbox("Choose an indicator", list(opts), key="overview_indicator")
                col, unit, explanation = opts[selected]
                d = data_for(features, selected, col)
                if not d.empty:
                    fig = px.line(d, x="Year", y=selected, markers=True)
                    fig.update_traces(line={"color": C["accent"], "width": 2.7}, marker={"size": 6})
                    fig.update_yaxes(title=unit)
                    st.plotly_chart(style(fig, 360), use_container_width=True, key="overview_indicator_chart")
                    metric_cols = st.columns([1, 1, 2])
                    metric_cols[0].metric("Latest value", fmt(float(d.iloc[-1][selected]), f" {unit}"))
                    metric_cols[1].metric("Latest year", str(int(d.iloc[-1].Year)))
                    metric_cols[2].caption(explanation)
                    st.button("View full trend →", key="view_full_trend", use_container_width=True,
                              on_click=lambda p="Economic Trends", ind=selected: st.session_state.update({"page": p, "trend_indicator": ind}))

elif page=="Economic Trends":
    st.markdown('<div class="kicker">02 / ECONOMIC TRENDS</div>',unsafe_allow_html=True); st.title("Explore the indicators")
    st.caption("Choose an indicator and time window. The chart, figures, table, and download update together.")
    opts=available(features)
    if not opts: st.warning("No recognized indicator columns were found. Check the processed CSV and its column names.")
    elif features is None or features.empty or "Year" not in features: st.info("The processed dataset or Year column is unavailable.")
    else:
        labels=list(opts); current=st.session_state.trend_indicator
        if current not in labels: current=labels[0]
        selected=st.selectbox("Indicator",labels,index=labels.index(current),key="trend_select")
        st.session_state.trend_indicator=selected
        col,unit,explanation=opts[selected]
        lo,hi=int(features.Year.min()),int(features.Year.max())
        years=st.slider("Historical period",lo,hi,(max(lo,hi-20),hi),key="trend_year_range")
        d=data_for(features,selected,col,years[0],years[1])
        if d.empty: st.info("No observations in this period.")
        else:
            last=d.iloc[-1]; m=st.columns(4)
            m[0].metric("Latest in range",fmt(float(last[selected]),f" {unit}")); m[1].metric("Year",str(int(last.Year))); m[2].metric("Average",fmt(float(d[selected].mean()),f" {unit}")); m[3].metric("Observations",str(len(d)))
            with st.container(border=True):
                st.markdown(f"#### {selected}"); st.caption(explanation)
                fig=px.line(d,x="Year",y=selected,markers=True); fig.update_traces(line={"color":C["accent"],"width":3},marker={"size":6},name="Historical")
                display_table=d.copy()
                if selected == "GDP Growth":
                    forecast_target = int(st.session_state.get("last_valid_forecast_year", int(features.Year.max()) + 1))
                    last_observed_year = int(features.Year.max())
                    if forecast_target > last_observed_year:
                        path, path_status = forecast_path(forecast_target)
                        if not path.empty:
                            pred=path.rename(columns={"Predicted GDP growth (%)":selected})
                            latest_observed = data_for(features, selected, col).tail(1)
                            pred_line=pd.concat([latest_observed,pred],ignore_index=True)
                            fig.add_scatter(x=pred_line["Year"],y=pred_line[selected],mode="lines+markers",name="Forecast / scenario",line={"color":"#D97706" if not DARK else "#FBBF24","width":3,"dash":"dash"},marker={"size":7})
                            display_table=pd.concat([d,pred],ignore_index=True)
                            st.caption(f"GDP forecast/scenario shown through {forecast_target}. Status: {path_status}.")
                        else:
                            st.warning(path_status)
                    else:
                        st.caption(f"The selected target year {forecast_target} is historical. This page shows observed data; no forecast line is added.")
                fig.update_yaxes(title=unit)
                st.plotly_chart(style(fig,435),use_container_width=True,key="trend_chart")
            with st.container(border=True):
                st.markdown("#### Observations and forecast" if selected == "GDP Growth" else "#### Historical observations")
                show_table(display_table)
                st.download_button("Download filtered data CSV",display_table.to_csv(index=False).encode("utf-8"),f"{col.lower()}_{years[0]}_{years[1]}.csv","text/csv",key="trend_csv")

elif page=="Country Comparison":
    st.markdown('<div class="kicker">03 / GLOBAL COMPARISON</div>',unsafe_allow_html=True); st.title("Compare economies")
    st.caption("Historical World Bank observations only; this section does not forecast other countries.")
    with st.container(border=True):
        x,y=st.columns(2)
        start=x.number_input("From year",1960,2025,2010,1,key="country_start"); end=y.number_input("To year",1960,2025,2025,1,key="country_end")
        indicator=st.selectbox("Indicator",list(WB),key="country_indicator")
        chosen=st.multiselect("Countries",list(COUNTRIES),default=list(COUNTRIES),key="country_choices")
        if start>end: st.error("Start year must not be later than end year.")
        elif st.button("Compare selected economies",type="primary",key="compare_button"):
            if not chosen: st.warning("Select at least one country.")
            else:
                try:
                    with st.spinner("Retrieving World Bank observations…"):
                        result=world_bank(WB[indicator],tuple(COUNTRIES[c] for c in chosen),int(start),int(end))
                    st.session_state.country_result={"data":result.to_dict("records"),"indicator":indicator,"start":int(start),"end":int(end)}
                    st.session_state.country_error=None
                except Exception as exc:
                    st.session_state.country_error=f"World Bank data could not be retrieved. Check your connection and retry. ({type(exc).__name__})"
                    st.session_state.country_result=None
    if st.session_state.country_error: st.error(st.session_state.country_error)
    saved=st.session_state.country_result
    if saved:
        d=pd.DataFrame(saved["data"],columns=["Country","Year","Value"])
        if d.empty: st.info("No observations returned for this selection.")
        else:
            with st.container(border=True):
                st.markdown(f"#### {saved['indicator']} · {saved['start']}–{saved['end']}")
                fig=px.line(d,x="Year",y="Value",color="Country",markers=True); fig.update_yaxes(title=saved["indicator"])
                st.plotly_chart(style(fig,435),use_container_width=True,key="country_chart")
            latest=d.sort_values("Year").groupby("Country",as_index=False).tail(1).sort_values("Value",ascending=False)
            with st.container(border=True):
                st.markdown("#### Latest available values"); show_table(latest)
                st.download_button("Download comparison CSV",d.to_csv(index=False).encode("utf-8"),"country_comparison.csv","text/csv",key="country_csv")

elif page=="Model Performance":
    st.markdown('<div class="kicker">04 / MODEL EVALUATION</div>',unsafe_allow_html=True); st.title("Understand model performance")
    st.caption("Recorded project evaluation results. These do not establish that one model is universally best.")
    with st.container(border=True):
        show_table(METRICS.style.format({"Mean MAE":"{:.3f}","Mean RMSE":"{:.3f}","Mean R²":"{:.3f}"}))
    metric=st.selectbox("Compare metric",["Mean MAE","Mean RMSE","Mean R²"],key="model_metric")
    d=METRICS.sort_values(metric,ascending=(metric!="Mean R²"))
    fig=px.bar(d,x="Model",y=metric,color="Model",text_auto=".3f"); fig.update_layout(showlegend=False,xaxis_title=None,yaxis_title=metric)
    st.plotly_chart(style(fig,405),use_container_width=True,key="model_chart")
    with st.expander("What do these metrics mean?"):
        st.markdown("- **MAE:** average absolute prediction error; lower is better.\n- **RMSE:** penalizes larger errors more strongly; lower is better.\n- **R²:** negative values indicate worse performance than a mean-based reference.")
    st.warning("Cross-validation favored the naive baseline overall (MAE 1.993), while the recorded holdout test favored XGBoost on MAE (3.197 vs 4.394). All listed cross-validation R² values are negative; do not describe XGBoost as consistently superior.")
    st.markdown("#### Recorded holdout MAE")
    show_table(pd.DataFrame([{"Model":"XGBoost","Holdout MAE":3.197},{"Model":"Naive Baseline","Holdout MAE":4.394}]))

elif page=="Data & Downloads":
    st.markdown('<div class="kicker">05 / DATA LIBRARY</div>',unsafe_allow_html=True); st.title("Data & downloads")
    st.caption("Preview and export saved project data without changing source files.")
    if features is not None:
        with st.container(border=True):
            st.markdown("#### Processed economic dataset"); st.caption(f"{len(features):,} rows · {len(features.columns):,} columns")
            show_table(features)
            st.download_button("Download processed dataset CSV",features.to_csv(index=False).encode("utf-8"),"india_economic_growth_features.csv","text/csv",key="features_csv")
    else: st.info("Processed dataset unavailable.")
    if forecast is not None:
        with st.container(border=True):
            st.markdown("#### Saved GDP forecast"); show_table(forecast)
            st.download_button("Download forecast CSV",forecast.to_csv(index=False).encode("utf-8"),"india_gdp_forecast.csv","text/csv",key="forecast_csv")
    else: st.info("Forecast file unavailable.")
    st.caption("Historical values come from the saved CSV. Forecast values are model-generated estimates, not official projections.")
