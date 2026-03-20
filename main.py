import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (MUST be first Streamlit command)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ GreenShift AI — Renewable Energy Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":    "https://github.com",
        "Report a bug": None,
        "About":       "**GreenShift AI** — Renewable Energy Adoption Predictor powered by ML",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (dark-green biopunk aesthetic)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Fonts ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ──────────────────────────────── */
:root {
  --bg-0:       #050d06;
  --bg-1:       #091410;
  --bg-2:       #0f1f19;
  --bg-3:       #162b22;
  --card:       #0d1e17;
  --border:     #1d3d2e;
  --g1:         #00ff87;
  --g2:         #00c96b;
  --g3:         #00954e;
  --b1:         #38bdf8;
  --amber:      #fbbf24;
  --red:        #f87171;
  --txt-p:      #e8f5ef;
  --txt-s:      #8db8a2;
  --radius:     14px;
}

/* ── Base Reset ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-0) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--txt-p);
}
[data-testid="stSidebar"] {
  background: var(--bg-1) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="block-container"] { padding-top: 1rem; padding-bottom: 3rem; }

/* ── Hide default Streamlit elements ─────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Headings ────────────────────────────────────── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

/* ── Metric cards ────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.25rem !important;
}
[data-testid="stMetricValue"] {
  color: var(--g1) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
  color: var(--txt-s) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] > div { font-size: 0.8rem !important; }

/* ── Sliders ─────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div {
  color: var(--g1) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--g1) !important;
  border-color: var(--g1) !important;
}

/* ── Select / Number inputs ─────────────────────── */
[data-testid="stSelectbox"] select,
[data-testid="stNumberInput"] input {
  background: var(--bg-2) !important;
  border: 1px solid var(--border) !important;
  color: var(--txt-p) !important;
  border-radius: 8px !important;
}

/* ── Buttons ─────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, var(--g3), var(--g2)) !important;
  color: #050d06 !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.55rem 1.5rem !important;
  font-size: 0.9rem !important;
  transition: all 0.2s !important;
  box-shadow: 0 0 18px rgba(0,255,135,0.3) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 28px rgba(0,255,135,0.5) !important;
}

/* ── Tabs ────────────────────────────────────────── */
[data-baseweb="tab-list"] {
  background: var(--bg-2) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 4px !important;
}
[data-baseweb="tab"] {
  color: var(--txt-s) !important;
  font-family: 'Syne', sans-serif !important;
  border-radius: 8px !important;
}
[aria-selected="true"] {
  background: var(--bg-3) !important;
  color: var(--g1) !important;
}

/* ── Progress bar ────────────────────────────────── */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--g3), var(--g1)) !important;
}

/* ── Expander ────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* ── Dataframe ────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: var(--radius) !important; }

/* ── Divider ─────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Custom cards ────────────────────────────────── */
.gs-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
}
.gs-card-glow {
  background: var(--card);
  border: 1px solid var(--g3);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 0 20px rgba(0,149,78,0.15);
}
.gs-badge {
  display: inline-block;
  padding: 0.2rem 0.7rem;
  border-radius: 99px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-family: 'Syne', sans-serif;
}
.gs-badge-green  { background: rgba(0,255,135,0.15); color: var(--g1); border: 1px solid var(--g3); }
.gs-badge-blue   { background: rgba(56,189,248,0.12); color: var(--b1); border: 1px solid #1e6fa0; }
.gs-badge-amber  { background: rgba(251,191,36,0.12); color: var(--amber); border: 1px solid #7a5a0e; }
.gs-badge-red    { background: rgba(248,113,113,0.12); color: var(--red); border: 1px solid #7a2020; }

.prediction-hero {
  text-align: center;
  padding: 2.5rem 1rem;
  background: var(--card);
  border-radius: var(--radius);
  border: 1px solid var(--border);
}
.prediction-hero .prob-number {
  font-family: 'Syne', sans-serif;
  font-size: 5rem;
  font-weight: 800;
  line-height: 1;
}
.prediction-hero .verdict-text {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  margin-top: 0.5rem;
}

.sidebar-logo {
  text-align: center;
  padding: 1.5rem 0 1rem;
}
.sidebar-logo .logo-icon {
  font-size: 3rem;
  animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { transform: scale(1); filter: drop-shadow(0 0 0 rgba(0,255,135,0)); }
  50%       { transform: scale(1.05); filter: drop-shadow(0 0 12px rgba(0,255,135,0.7)); }
}
.sidebar-logo .logo-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--g1), var(--b1));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-top: 0.4rem;
}
.sidebar-logo .logo-sub {
  color: var(--txt-s);
  font-size: 0.75rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

/* ── Hero Banner ─────────────────────────────────── */
.hero-banner {
  background: linear-gradient(135deg, var(--bg-2) 0%, var(--bg-3) 100%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(0,255,135,0.12) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.4rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--g1) 0%, var(--b1) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
}
.hero-sub {
  color: var(--txt-s);
  font-size: 1rem;
  margin-top: 0.5rem;
  max-width: 560px;
}

/* ── Stat row ────────────────────────────────────── */
.stat-row {
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 1.5rem;
}
.stat-chip {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--bg-3);
  border: 1px solid var(--border);
  border-radius: 99px;
  padding: 0.35rem 0.9rem;
  font-size: 0.8rem;
  color: var(--txt-s);
}
