import html
import json
import re
import sys
import textwrap
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama
except ImportError:  # chatbot degrades gracefully instead of crashing the app
    ollama = None


# ============================================================
# PAGE CONFIG  (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="SupportAI | Customer Support Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
LLM_MODEL = "llama3.2:3b"

NAV_SECTIONS = [
    (
        "WORKSPACE",
        [
            ("📊", "Complaint Analysis"),
            ("💬", "AI Support Assistant"),
            ("🧠", "ML Intelligence"),
        ],
    ),
    (
        "INTELLIGENCE",
        [
            ("⚡", "Semantic Issue Detection"),
            ("🎯", "Priority Prediction"),
            ("🚨", "Escalation Risk Engine"),
        ],
    ),
]

RESOLUTION_TESTS = [
    ("My phone battery drains very quickly and shuts down at 30%.", {"Battery life"}),
    ("My order has not arrived yet and the tracking has not updated.", {"Delivery problem"}),
    ("I was charged twice for the same order.", {"Payment issue"}),
    ("I want to cancel my order before it ships.", {"Cancellation request"}),
    ("I can't log in to my account anymore.", {"Account access"}),
    ("My account has been hacked and unauthorized transactions were made.", {"Account access"}),
    ("My files disappeared after the update and I lost important data.", {"Data loss"}),
    ("My Wi-Fi keeps disconnecting every few minutes.", {"Network problem"}),
    ("The software keeps crashing whenever I open it.", {"Software bug"}),
    ("The screen is not working and stays black.", {"Display issue", "Hardware issue"}),
]


SAMPLE_COMPLAINTS = {
    "🔋 Battery": "My phone battery drains very quickly and the device shuts down even when it shows 30% battery.",
    "🔐 Hacked": "My account has been hacked and I cannot log in. Someone made unauthorized purchases. This is urgent.",
    "💳 Payment": "I was charged twice for the same order and I want my money back immediately.",
    "📦 Delivery": "My order was supposed to arrive last week and the tracking has not updated for days.",
}


# ============================================================
# HELPERS
# ============================================================

def render_html(content):
    st.html(textwrap.dedent(content).strip())


def esc(value):
    return html.escape(str(value))


def spacer(px=18):
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


def page_header(title, description):
    render_html(
        f"""
        <div class="page-title">{esc(title)}</div>
        <div class="page-description">{esc(description)}</div>
        """
    )


ACCENTS = {
    "issue-value": "#64a8ff",
    "critical-value": "#ff6574", "risk-critical": "#ff6574",
    "high-value": "#ff9f5b", "risk-high": "#ff9f5b",
    "medium-value": "#f4c95d", "risk-medium": "#f4c95d",
    "low-value": "#5cdd9a", "risk-low": "#5cdd9a",
}


def result_card(label, value, small="", value_class=""):
    accent = ACCENTS.get(value_class, "#5a7cff")
    return f"""
        <div class="result-card" style="--accent:{accent};">
            <div class="result-label">{esc(label)}</div>
            <div class="result-value {value_class}">{esc(value)}</div>
            <div class="result-small">{esc(small)}</div>
        </div>
    """


def info_card(title, body, value=None):
    value_html = f'<div class="card-value">{esc(value)}</div>' if value else ""
    return f"""
        <div class="glass-card">
            <div class="card-title">{esc(title)}</div>
            {value_html}
            <div class="card-description">{body}</div>
        </div>
    """


def pipeline_html(steps):
    parts = []
    for i, step in enumerate(steps):
        if i:
            parts.append('<div class="pipeline-arrow">→</div>')
        parts.append(f'<div class="pipeline-step">{esc(step)}</div>')
    return f'<div class="pipeline">{"".join(parts)}</div>'


PRIORITY_CLASS = {
    "Critical": "critical-value",
    "High": "high-value",
    "Medium": "medium-value",
    "Low": "low-value",
}

RISK_CLASS = {
    "Critical": "risk-critical",
    "High": "risk-high",
    "Medium": "risk-medium",
    "Low": "risk-low",
}


# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, button, textarea, input {
        font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
    }

    /* ---------------- GLOBAL ---------------- */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(53, 88, 255, 0.10), transparent 28%),
            radial-gradient(circle at 85% 20%, rgba(125, 70, 255, 0.08), transparent 25%),
            #070b14;
        color: #f4f7ff;
    }

    /* Keep the header (it holds the sidebar open/close arrow) but make it invisible */
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stDeployButton"], [data-testid="stAppDeployButton"], .stAppDeployButton,
    [data-testid="stMainMenu"], #MainMenu { display: none !important; }
    footer { visibility: hidden; }

    /* sidebar collapse / expand arrows stay visible and themed */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] button {
        color: #9fb2d6 !important;
    }

    .block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1500px; }

    /* ---------------- SIDEBAR ---------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1120 0%, #090d18 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
        min-width: 290px;
    }
    section[data-testid="stSidebar"] > div { padding-top: 0.6rem; }

    .sidebar-brand {
        padding: 18px 16px 20px 16px;
        margin-bottom: 8px;
        border: 1px solid rgba(87,116,255,0.18);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(44,64,126,0.20), rgba(18,24,43,0.65));
        box-shadow: 0 12px 35px rgba(0,0,0,0.22);
    }
    .sidebar-logo {
        width: 44px; height: 44px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 13px;
        background: linear-gradient(135deg, #3b82f6, #7c3aed);
        font-size: 21px; margin-bottom: 12px;
        box-shadow: 0 8px 25px rgba(73,91,255,0.30);
    }
    .sidebar-title { font-size: 21px; font-weight: 800; color: #fff; letter-spacing: -0.4px; }
    .sidebar-subtitle { margin-top: 4px; font-size: 11px; color: #7f91b5; line-height: 1.4; }
    .sidebar-section {
        margin: 22px 6px 8px 6px;
        font-size: 10px; font-weight: 800; letter-spacing: 1.5px;
        color: #617397; text-transform: uppercase;
    }

    /* navigation buttons */
    [data-testid="stSidebar"] .stButton { margin-bottom: 4px; }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 44px;
        border-radius: 11px;
        border: 1px solid transparent;
        background: transparent;
        color: #93a4c4;
        font-size: 13px;
        font-weight: 600;
        justify-content: flex-start;
        padding-left: 14px;
        transition: all 0.18s ease;
    }
    [data-testid="stSidebar"] .stButton > button > div,
    [data-testid="stSidebar"] .stButton > button p {
        justify-content: flex-start !important;
        text-align: left !important;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(65,91,160,0.15);
        border-color: rgba(91,117,255,0.22);
        color: #ffffff;
    }
    /* ACTIVE page = primary button (old + new Streamlit selectors) */
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, rgba(50,101,255,0.30), rgba(113,68,226,0.22));
        border: 1px solid rgba(77,125,255,0.45);
        color: #ffffff;
        box-shadow: 0 8px 22px rgba(31,59,130,0.22);
    }

    .sidebar-stat {
        display: flex; justify-content: space-between;
        padding: 9px 12px; margin-top: 6px;
        border-radius: 10px;
        background: rgba(65,91,150,0.10);
        border: 1px solid rgba(84,111,179,0.12);
        color: #8ea1c4; font-size: 11px;
    }
    .sidebar-stat b { color: #fff; }
    .sidebar-footer {
        margin-top: 26px; padding: 12px;
        border-top: 1px solid rgba(255,255,255,0.06);
        color: #566786; font-size: 10px; line-height: 1.6; text-align: center;
    }

    /* ---------------- TOP HEADER ---------------- */
    .top-header {
        display: flex; justify-content: space-between; align-items: center; gap: 20px;
        padding: 24px 26px; border-radius: 20px;
        border: 1px solid rgba(93,117,184,0.16);
        background: linear-gradient(135deg, rgba(23,29,47,0.96), rgba(14,19,33,0.94));
        box-shadow: 0 18px 50px rgba(0,0,0,0.20);
        margin-bottom: 24px;
    }
    .brand-title { font-size: 29px; font-weight: 800; letter-spacing: -1px; color: #fff; line-height: 1.15; }
    .brand-title span {
        background: linear-gradient(90deg, #54a8ff, #9b6cff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .brand-subtitle { margin-top: 7px; color: #7184a8; font-size: 12px; }
    .header-badges { display: flex; gap: 10px; flex-wrap: wrap; }
    .system-status {
        display: flex; align-items: center; gap: 10px;
        padding: 11px 15px; border-radius: 13px; white-space: nowrap;
        font-size: 11px; font-weight: 600;
    }
    .status-online { border: 1px solid rgba(77,210,151,0.16); background: rgba(34,109,77,0.10); color: #9be6c1; }
    .status-offline { border: 1px solid rgba(255,110,130,0.20); background: rgba(120,40,55,0.12); color: #ff9aa6; }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; }
    .status-online .status-dot { background: #48d597; box-shadow: 0 0 12px rgba(72,213,151,0.8); }
    .status-offline .status-dot { background: #ff6574; box-shadow: 0 0 12px rgba(255,101,116,0.8); }

    /* ---------------- PAGE TITLES ---------------- */
    .page-title { font-size: 22px; font-weight: 800; color: #f8faff; margin-bottom: 5px; }
    .page-description { font-size: 12px; color: #7183a7; margin-bottom: 20px; line-height: 1.6; }

    /* ---------------- CARDS ---------------- */
    .glass-card {
        border: 1px solid rgba(86,110,170,0.18); border-radius: 18px;
        background: linear-gradient(145deg, rgba(19,26,43,0.95), rgba(12,17,29,0.98));
        padding: 20px; box-shadow: 0 15px 45px rgba(0,0,0,0.18);
        height: 100%;
    }
    .card-title {
        font-size: 11px; font-weight: 800; color: #8fa4ca;
        letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 10px;
    }
    .card-value { font-size: 23px; font-weight: 800; color: #fff; }
    .card-description { margin-top: 6px; font-size: 12px; color: #7e91b0; line-height: 1.65; }

    .section-label {
        color: #8da2c8; font-size: 10px; font-weight: 800;
        letter-spacing: 1.4px; margin: 4px 0 9px 0; text-transform: uppercase;
    }

    /* ---------------- INPUTS ---------------- */
    textarea {
        background: #0a0f1a !important; color: #f4f7ff !important;
        border: 1px solid #253452 !important; border-radius: 13px !important;
    }
    textarea:focus { border-color: #4f7cff !important; box-shadow: 0 0 0 1px #4f7cff !important; }

    .stButton > button, .stDownloadButton > button { border-radius: 11px; font-weight: 700; transition: all 0.18s ease; min-height: 46px; }

    /* main-area (non-sidebar) buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #3b6cff, #7040e0);
        border: none; color: #fff; min-height: 46px;
        box-shadow: 0 10px 28px rgba(60,90,255,0.30);
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        filter: brightness(1.12); transform: translateY(-1px);
        box-shadow: 0 14px 34px rgba(80,100,255,0.42);
    }
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"],
    .stDownloadButton > button {
        background: rgba(30,42,72,0.55); border: 1px solid rgba(86,110,170,0.28); color: #b8c7e6;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover,
    .stDownloadButton > button:hover {
        border-color: rgba(100,140,255,0.55); color: #fff; background: rgba(50,75,140,0.35);
    }

    /* ---------------- RESULT CARDS ---------------- */
    /* Fixed-height cards: every card in a row is exactly the same size */
    .result-card {
        --accent: #5a7cff;
        position: relative; overflow: hidden;
        box-sizing: border-box;
        height: 172px; padding: 18px 18px 16px 18px; border-radius: 17px;
        display: flex; flex-direction: column;
        border: 1px solid rgba(89,113,172,0.18);
        background:
            radial-gradient(circle at 100% 0%, color-mix(in srgb, var(--accent) 14%, transparent), transparent 55%),
            linear-gradient(145deg, rgba(20,27,45,0.98), rgba(12,17,29,0.98));
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card::before {
        content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--accent), transparent);
    }
    .result-card:hover {
        transform: translateY(-3px);
        border-color: color-mix(in srgb, var(--accent) 45%, transparent);
        box-shadow: 0 18px 40px rgba(0,0,0,0.30), 0 0 0 1px color-mix(in srgb, var(--accent) 20%, transparent);
    }
    .result-label {
        color: #7083a5; font-size: 10px; font-weight: 800;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 10px;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .result-value {
        color: #fff; font-size: 18px; font-weight: 800; line-height: 1.25;
        display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
    }
    .result-small {
        margin-top: auto; padding-top: 8px; color: #7184a8; font-size: 11px; line-height: 1.45;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }

    .issue-value { color: #64a8ff; }
    .critical-value, .risk-critical { color: #ff6574; }
    .high-value, .risk-high { color: #ff9f5b; }
    .medium-value, .risk-medium { color: #f4c95d; }
    .low-value, .risk-low { color: #5cdd9a; }

    /* ---------------- RESOLUTION / ESCALATION ---------------- */
    .resolution-card {
        border: 1px solid rgba(74,142,255,0.20); border-radius: 17px; padding: 20px;
        background: linear-gradient(145deg, rgba(28,45,80,0.34), rgba(13,20,35,0.96));
    }
    .resolution-title {
        color: #72aaff; font-size: 11px; font-weight: 800;
        letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 10px;
    }
    .resolution-text { color: #d9e2f4; font-size: 13px; line-height: 1.75; }

    .escalation-card {
        border-radius: 17px; padding: 20px;
        border: 1px solid rgba(255,110,130,0.16);
        background: linear-gradient(145deg, rgba(65,27,43,0.24), rgba(14,18,30,0.96));
    }
    .risk-score { font-size: 28px; font-weight: 800; color: #fff; }
    .reason-item {
        padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #9aaccc; font-size: 12px;
    }
    .reason-item:last-child { border-bottom: none; }

    /* similarity bars */
    .match-row { margin: 10px 0; }
    .match-head { display: flex; justify-content: space-between; font-size: 12px; color: #b4c3e0; margin-bottom: 5px; }
    .match-bar { height: 6px; border-radius: 6px; background: rgba(255,255,255,0.06); overflow: hidden; }
    .match-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); }

    /* history table */
    .history-row {
        display: grid; grid-template-columns: 70px 1fr 150px 90px 90px; gap: 12px;
        padding: 10px 4px; border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 12px; color: #9aaccc; align-items: center;
    }
    .history-row.head { font-size: 10px; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase; color: #617397; }
    .history-row:last-child { border-bottom: none; }
    .history-text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    /* ---------------- CHAT ---------------- */
    .chat-header {
        display: flex; align-items: center; gap: 12px;
        padding: 3px 0 17px 0; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 15px;
    }
    .chat-avatar {
        width: 43px; height: 43px; border-radius: 13px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #3b82f6, #7c3aed);
        box-shadow: 0 8px 25px rgba(74,95,255,0.25); font-size: 20px;
    }
    .chat-name { color: #fff; font-weight: 800; font-size: 14px; }
    .chat-status { margin-top: 4px; color: #69dba3; font-size: 10px; }
    .chat-welcome {
        padding: 13px; border-radius: 13px;
        background: rgba(48,66,112,0.16); border: 1px solid rgba(87,115,181,0.14);
        color: #94a7c8; font-size: 12px; line-height: 1.6; margin-bottom: 12px;
    }

    /* ---------------- PIPELINE / TAGS ---------------- */
    .pipeline { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 15px; }
    .pipeline-step {
        padding: 8px 12px; border-radius: 9px;
        background: rgba(65,91,150,0.14); border: 1px solid rgba(84,111,179,0.15);
        color: #9badcc; font-size: 11px; font-weight: 700;
    }
    .pipeline-arrow { color: #4d70aa; font-size: 12px; }
    .tag {
        display: inline-block; padding: 6px 11px; margin: 3px; border-radius: 8px;
        background: rgba(69,95,160,0.14); border: 1px solid rgba(86,112,179,0.16);
        color: #9eb0ce; font-size: 11px;
    }

    .app-footer {
        margin-top: 35px; padding-top: 16px;
        border-top: 1px solid rgba(255,255,255,0.06);
        color: #4f607d; text-align: center; font-size: 10px;
    }

    /* ================= POLISH LAYER ================= */

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0%   { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.45; }
    }

    .result-card, .glass-card, .resolution-card, .escalation-card, .top-header, .kpi {
        animation: fadeUp 0.45s ease both;
    }
    .status-dot { animation: pulse 2s ease-in-out infinite; }

    /* header: gradient hairline on top */
    .top-header { position: relative; overflow: hidden; }
    .top-header::before {
        content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6);
        background-size: 200% 100%; animation: shimmer 6s linear infinite;
    }
    .brand-title span {
        background: linear-gradient(90deg, #54a8ff, #9b6cff, #ff7ad9, #54a8ff);
        background-size: 200% 100%; animation: shimmer 6s linear infinite;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* glass cards: soft hover */
    .glass-card { transition: border-color 0.2s ease, transform 0.2s ease; }
    .glass-card:hover { border-color: rgba(110,140,230,0.35); }

    /* KPI strip */
    .kpi {
        display: flex; align-items: center; gap: 14px;
        padding: 16px 18px; border-radius: 16px;
        border: 1px solid rgba(86,110,170,0.16);
        background: linear-gradient(145deg, rgba(19,26,43,0.9), rgba(12,17,29,0.95));
    }
    .kpi-icon {
        width: 42px; height: 42px; flex: 0 0 42px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center; font-size: 19px;
        background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(139,92,246,0.22));
        border: 1px solid rgba(110,140,255,0.25);
    }
    .kpi-value { font-size: 22px; font-weight: 800; color: #fff; line-height: 1.1; }
    .kpi-label { font-size: 10px; font-weight: 700; letter-spacing: 1.1px; text-transform: uppercase; color: #6f82a4; margin-top: 3px; }

    /* sidebar active item gets an accent bar */
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
        box-shadow: inset 3px 0 0 #5b8cff, 0 8px 22px rgba(31,59,130,0.22);
    }

    /* chat bubbles + avatars (replaces default red / orange) */
    [data-testid="stChatMessage"] {
        background: rgba(30,42,72,0.22);
        border: 1px solid rgba(86,110,170,0.12);
        border-radius: 14px; padding: 10px 12px;
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important; color: #fff !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important; color: #fff !important;
    }
    [data-testid="stChatInput"] { border-radius: 14px; }

    /* text-area focus ring + scrollbars */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: #243252; border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: #34477a; }

    /* bordered containers (chat panel) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(86,110,170,0.20) !important;
        border-radius: 18px !important;
        background: linear-gradient(145deg, rgba(19,26,43,0.55), rgba(12,17,29,0.7));
    }

    /* ---------- resolution card extras ---------- */
    .res-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
    .level-badge {
        padding: 4px 10px; border-radius: 999px; font-size: 10px; font-weight: 800;
        letter-spacing: 0.8px; text-transform: uppercase; white-space: nowrap;
    }
    .lvl-standard { color: #5cdd9a; background: rgba(92,221,154,0.10); border: 1px solid rgba(92,221,154,0.28); }
    .lvl-elevated { color: #ff9f5b; background: rgba(255,159,91,0.10); border: 1px solid rgba(255,159,91,0.30); }
    .lvl-critical { color: #ff6574; background: rgba(255,101,116,0.12); border: 1px solid rgba(255,101,116,0.34); }
    .res-footer {
        display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
        margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.07);
    }
    .res-meta-label { font-size: 10px; font-weight: 800; letter-spacing: 1.1px; text-transform: uppercase; color: #6f82a4; margin-bottom: 5px; }
    .res-meta-value { font-size: 14px; font-weight: 700; color: #e6edfb; }
    .res-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .res-chip {
        padding: 6px 11px; border-radius: 9px; font-size: 11px; color: #8fa2c6;
        background: rgba(65,91,150,0.12); border: 1px solid rgba(84,111,179,0.16);
    }
    .res-chip b { font-weight: 800; }

    /* test lab */
    .test-row {
        display: grid; grid-template-columns: 1.6fr 1fr 1fr 80px 80px 70px; gap: 12px;
        padding: 10px 4px; border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 12px; color: #9aaccc; align-items: center;
    }
    .test-row.head { font-size: 10px; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase; color: #617397; }
    .test-row:last-child { border-bottom: none; }

    /* ---------- escalation: hero, meter, breakdown ---------- */
    .risk-hero { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin: 4px 0 18px 0; }
    .risk-hero .risk-score { font-size: 36px; letter-spacing: -0.5px; }
    .risk-action { font-size: 12px; color: #8da2c8; font-weight: 600; }

    .meter-wrap { margin-top: 4px; padding-bottom: 4px; }
    .meter { position: relative; display: flex; height: 10px; gap: 2px; }
    .meter-seg { height: 100%; }
    .meter-seg:nth-child(1) { border-radius: 10px 0 0 10px; }
    .meter-seg:nth-child(4) { border-radius: 0 10px 10px 0; }
    .meter-low { background: rgba(92,221,154,0.75); }
    .meter-medium { background: rgba(244,201,93,0.75); }
    .meter-high { background: rgba(255,159,91,0.80); }
    .meter-critical { background: rgba(255,101,116,0.85); }
    .meter-marker {
        position: absolute; top: -5px; width: 4px; height: 20px; border-radius: 3px;
        background: #fff; transform: translateX(-50%);
        box-shadow: 0 0 14px rgba(255,255,255,0.85);
    }
    .meter-scale { position: relative; height: 14px; margin-top: 10px; font-size: 10px; color: #6f82a4; }
    .meter-scale span { position: absolute; transform: translateX(-50%); }
    .meter-scale span:first-child { transform: none; }
    .meter-scale span:last-child { transform: translateX(-100%); }

    .bd-row {
        display: grid; grid-template-columns: 1fr 110px 38px; gap: 12px; align-items: center;
        padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 12px; color: #aebcd9;
    }
    .bd-row:last-child { border-bottom: none; }
    .bd-bar { height: 6px; border-radius: 6px; background: rgba(255,255,255,0.06); overflow: hidden; }
    .bd-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #ff9f5b, #ff6574); }
    .bd-pts { text-align: right; font-weight: 800; color: #fff; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 14px 0; }

    /* ---------- scoring rules ---------- */
    .rule-pills { display: flex; flex-wrap: wrap; gap: 8px; }
    .rule-pill {
        padding: 6px 11px; border-radius: 9px; font-size: 11px; font-weight: 600; color: #9eb0ce;
        background: rgba(69,95,160,0.14); border: 1px solid rgba(86,112,179,0.16);
    }
    .rule-pill b { color: #fff; font-weight: 700; }
    .rule-row {
        display: flex; gap: 12px; padding: 7px 0; font-size: 12px; color: #9aaccc;
        border-bottom: 1px solid rgba(255,255,255,0.05); line-height: 1.5;
    }
    .rule-row:last-of-type { border-bottom: none; }
    .rule-pts { flex: 0 0 30px; font-weight: 800; color: #72aaff; }
    .rule-text b { color: #dbe4f5; font-weight: 700; }

    /* ---------- key / value card ---------- */
    .kv-row {
        display: flex; justify-content: space-between; gap: 16px; padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 12px;
    }
    .kv-row:last-child { border-bottom: none; }
    .kv-key { color: #7083a5; }
    .kv-val { color: #e6edfb; font-weight: 600; text-align: right; }

    /* ---------- ML pipeline grid ---------- */
    .stage-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
    .stage-card {
        position: relative; padding: 16px 14px; border-radius: 15px;
        border: 1px solid rgba(89,113,172,0.18);
        background: linear-gradient(145deg, rgba(20,27,45,0.98), rgba(12,17,29,0.98));
        transition: transform 0.2s ease, border-color 0.2s ease;
        animation: fadeUp 0.45s ease both;
    }
    .stage-card:hover { transform: translateY(-3px); border-color: rgba(100,140,255,0.45); }
    .stage-num { position: absolute; top: 11px; right: 13px; font-size: 10px; font-weight: 800; color: #4d70aa; }
    .stage-icon { font-size: 22px; margin-bottom: 10px; }
    .stage-title { font-size: 13px; font-weight: 800; color: #fff; }
    .stage-short { margin-top: 4px; font-size: 11px; color: #7184a8; line-height: 1.45; }

    /* ---------- tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.07); }
    .stTabs [data-baseweb="tab"] {
        height: 42px; padding: 0 14px; border-radius: 10px 10px 0 0;
        color: #7f91b5; font-weight: 600; font-size: 12px;
    }
    .stTabs [aria-selected="true"] { color: #ffffff !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }

    /* ---------- KPI v2 ---------- */
    .kpi { min-height: 92px; }
    .kpi-body { min-width: 0; }
    .kpi-label { margin: 0 0 3px 0; }
    .kpi-sub { margin-top: 4px; font-size: 11px; font-weight: 700; line-height: 1.3; }
    .sub-up { color: #5cdd9a; }
    .sub-down { color: #ff9f5b; }
    .sub-muted { color: #6f82a4; font-weight: 600; }

    /* ---------- insights ---------- */
    .stack-bar { display: flex; height: 12px; border-radius: 8px; overflow: hidden; gap: 2px; margin: 8px 0 14px 0; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px 16px; font-size: 11px; color: #8da2c8; }
    .legend-item i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .legend-item b { color: #fff; margin-left: 2px; }

    @media (max-width: 900px) {
        .top-header { flex-direction: column; align-items: flex-start; }
        .brand-title { font-size: 24px; }
        .history-row { grid-template-columns: 60px 1fr 90px; }
        .history-row > :nth-child(3), .history-row > :nth-child(5) { display: none; }
    }
    </style>
"""


@st.cache_resource
def minified_css(css):
    """Strip comments and extra whitespace once, so less CSS is sent on every rerun."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    css = re.sub(r"\s+", " ", css)
    return re.sub(r"\s*([{};,])\s*", r"\1", css).strip()


st.markdown(minified_css(CUSTOM_CSS), unsafe_allow_html=True)


# ============================================================
# LOAD MODEL ASSETS
# ============================================================

@st.cache_resource
def load_assets():
    issue_prototypes = joblib.load(MODEL_DIR / "issue_prototypes.pkl")
    severity_embeddings = joblib.load(MODEL_DIR / "severity_embeddings.pkl")
    resolution_map = joblib.load(MODEL_DIR / "resolution_map.pkl")
    return issue_prototypes, severity_embeddings, resolution_map


@st.cache_resource
def load_semantic_model():
    model = SentenceTransformer(str(MODEL_DIR / "semantic_model"))
    model.encode(["warm up"], convert_to_numpy=True)  # first encode is slow; do it once at startup
    return model


try:
    issue_prototypes, severity_embeddings, resolution_map = load_assets()
    semantic_model = load_semantic_model()
except Exception as e:
    st.error("Unable to load the project models. Please check the models folder.")
    st.code(str(e))
    st.stop()


@st.cache_data(ttl=30, show_spinner=False)
def ollama_online():
    """Lightweight check so the header shows the real assistant status."""
    if ollama is None:
        return False
    try:
        ollama.Client(timeout=2).list()
        return True
    except Exception:
        return False


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("page", "Complaint Analysis")
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("analysis_result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("complaint_input", "")
st.session_state.setdefault("esc_test_input", "")
st.session_state.setdefault("esc_test_result", None)

# Results saved by an older version of the app may lack newer fields -> reset them
_REQUIRED_FIELDS = {"resolution_team", "resolution_level", "top_matches", "timestamp", "pdf"}
_saved = st.session_state.analysis_result
if _saved and (
    not _REQUIRED_FIELDS <= _saved.keys() or "breakdown" not in _saved["escalation"]
):
    st.session_state.analysis_result = None
    st.session_state.history = []


def go_to(page_name):
    st.session_state.page = page_name


def load_sample(text):
    st.session_state.complaint_input = text


def clear_chat():
    st.session_state.chat_messages = []


def load_esc_sample(text):
    st.session_state.esc_test_input = text


# ============================================================
# ISSUE PREDICTION
# ============================================================

@st.cache_data(max_entries=512, show_spinner=False)
def embed(text):
    """Sentence embedding. Cached, so a complaint is only encoded once."""
    return semantic_model.encode([text], convert_to_numpy=True)


@st.cache_resource
def issue_index():
    """All issue prototypes stacked into one matrix (single similarity call)."""
    names = list(issue_prototypes.keys())
    matrix = np.vstack(
        [np.asarray(issue_prototypes[name]).reshape(1, -1) for name in names]
    )
    return names, matrix


def get_issue_scores(text):
    """Return [(issue, similarity), ...] sorted from best to worst match."""
    names, matrix = issue_index()
    similarities = cosine_similarity(embed(text), matrix)[0]
    return sorted(
        zip(names, (float(s) for s in similarities)),
        key=lambda item: item[1],
        reverse=True,
    )


def predict_issue_semantic(text):
    return get_issue_scores(text)[0][0]


# ============================================================
# PRIORITY PREDICTION
# ============================================================

CRITICAL_KEYWORDS = [
    "hacked", "compromised", "unauthorized", "security breach",
    "stolen data", "data loss", "fraud", "fraudulent",
    "account stolen", "identity theft",
]


def predict_priority(text):
    lowered = text.lower()

    # Critical-condition override
    if any(keyword in lowered for keyword in CRITICAL_KEYWORDS):
        return "Critical"

    # Semantic priority prediction
    embedding = embed(text)
    scores = {
        priority: cosine_similarity(embedding, examples).mean()
        for priority, examples in severity_embeddings.items()
    }
    return max(scores, key=scores.get)


# ============================================================
# RESOLUTION RECOMMENDATION
# ============================================================
#
# Resolution = f(issue, priority, escalation)
#   standard -> Low / Medium cases
#   elevated -> High priority OR High risk
#   critical -> Critical priority OR Critical risk (only where defined)
#

DEFAULT_RESOLUTION = (
    "Escalate this case to the appropriate support team for further assistance."
)

RESOLUTION_PLAYBOOK = {
    "Account access": {
        "team": "Account Security Team",
        "standard": (
            "Guide the customer through the official password-reset or account-recovery "
            "flow and verify their identity using approved methods only. Confirm the "
            "registered email or phone number and check whether the account is locked. "
            "Escalate to the account support team if recovery fails."
        ),
        "elevated": (
            "Initiate account recovery and password reset. Review recent login activity "
            "and revoke suspicious active sessions. If the customer still cannot access "
            "the account, escalate the case to the account security support team."
        ),
        "critical": (
            "Immediately secure the account by initiating the official account-recovery "
            "process and forcing a password reset. Revoke active sessions and review recent "
            "login activity for unauthorized access. If unauthorized transactions are "
            "detected, temporarily restrict affected payment methods and escalate the case "
            "to the security/fraud support team. Do not request passwords, OTPs, or other "
            "sensitive credentials through the support chat."
        ),
    },
    "Battery life": {
        "team": "Hardware Support",
        "standard": (
            "Check battery health and identify applications or processes causing excessive "
            "power consumption. Recommend reducing background activity, updating the device "
            "software, and checking power-management settings. If the battery is degraded, "
            "arrange a battery inspection."
        ),
        "elevated": (
            "Treat this as a possible device fault. Check battery health and recent "
            "power-drain causes, then arrange a hardware inspection or battery replacement. "
            "If the device overheats, swells or shuts down unexpectedly, advise the customer "
            "to stop charging and using it and escalate to hardware support immediately."
        ),
    },
    "Cancellation request": {
        "team": "Order Management Team",
        "standard": (
            "Verify the order status and cancellation eligibility. If the order has not "
            "been processed or shipped, initiate cancellation and confirm the applicable "
            "refund process."
        ),
        "elevated": (
            "Verify the order status immediately. If the order has not shipped, cancel it "
            "before dispatch and confirm the refund timeline. If it has already shipped, "
            "ask the delivery team to intercept it or arrange a return, and escalate if the "
            "customer is still being charged."
        ),
    },
    "Data loss": {
        "team": "Technical / Data Recovery Team",
        "standard": (
            "Identify the affected data and determine whether it can be recovered from "
            "available backups or recovery mechanisms. Avoid overwriting affected data and "
            "escalate the case to technical support when recovery is required."
        ),
        "elevated": (
            "Ask the customer to stop using the affected device or account so data is not "
            "overwritten. Identify what was lost and when, check for backups or recovery "
            "options, and escalate to technical support for recovery."
        ),
        "critical": (
            "Treat this as an urgent data-loss incident. Advise the customer to stop "
            "writing to the affected storage, establish the scope of the loss and the last "
            "known good backup, and escalate immediately to the technical/data-recovery "
            "team. Do not promise full recovery before the damage has been assessed."
        ),
    },
    "Delivery problem": {
        "team": "Delivery Support Team",
        "standard": (
            "Verify the order and shipment status, check the latest tracking information, "
            "and identify whether the package is delayed, lost, or incorrectly delivered. "
            "Escalate to the delivery team when the shipment cannot be located."
        ),
        "elevated": (
            "Verify the shipment status and latest tracking information immediately. If the "
            "package cannot be located or has exceeded the expected delivery period, "
            "escalate the case to the delivery support team for investigation."
        ),
    },
    "Display issue": {
        "team": "Hardware Support",
        "standard": (
            "Check display settings, connections, and device hardware. Restart the device "
            "and verify whether the issue persists. Escalate for hardware inspection if the "
            "display remains faulty."
        ),
        "elevated": (
            "Determine whether the fault is software or hardware (test with an external "
            "display, safe mode and cable checks). If the screen is blank, cracked or "
            "flickering, arrange a hardware inspection or replacement and escalate to "
            "hardware support."
        ),
    },
    "Hardware issue": {
        "team": "Hardware Support",
        "standard": (
            "Perform basic hardware diagnostics and identify the affected component. Check "
            "whether the issue is reproducible and recommend professional inspection or "
            "replacement if the hardware is faulty."
        ),
        "elevated": (
            "Run diagnostics to confirm the faulty component and check warranty status. "
            "Arrange repair or replacement, and escalate to hardware support right away if "
            "the device is unusable or poses a safety risk."
        ),
    },
    "Installation support": {
        "team": "Technical Support",
        "standard": (
            "Verify the installation requirements and guide the customer through the "
            "required installation steps. Check for dependency or configuration issues and "
            "escalate if installation continues to fail."
        ),
        "elevated": (
            "Collect the exact error message and system details, confirm the requirements, "
            "and guide the customer through a clean reinstall. If installation is blocking "
            "their work, escalate to technical support with the error details."
        ),
    },
    "Network problem": {
        "team": "Network / Technical Support",
        "standard": (
            "Check network connectivity, connection stability, and relevant network "
            "settings. Restart the connection, verify configuration, and escalate to "
            "technical support if the problem persists."
        ),
        "elevated": (
            "Check connectivity, router and device settings, and any known outage in the "
            "customer's area. Run a full network reset and collect error details. If the "
            "connection is completely down or affecting their work, escalate to "
            "network/technical support."
        ),
    },
    "Payment issue": {
        "team": "Payments / Fraud Team",
        "standard": (
            "Verify the transaction details, payment status, and billing record. Identify "
            "whether the issue is a failed, duplicate, or incorrect charge and guide the "
            "customer through the appropriate payment resolution process."
        ),
        "elevated": (
            "Verify the transaction and billing record immediately and identify whether the "
            "charge is failed, duplicate or incorrect. Start the correction or refund "
            "process, and escalate to the payments team if it cannot be resolved quickly."
        ),
        "critical": (
            "Verify the reported transaction and check for duplicate or unauthorized "
            "charges. Temporarily restrict the affected payment method if fraud is "
            "suspected and escalate the case to the payment/fraud support team for "
            "immediate investigation."
        ),
    },
    "Peripheral compatibility": {
        "team": "Technical Support",
        "standard": (
            "Verify the peripheral model, connection type, drivers, and supported device "
            "requirements. Update drivers or configuration and escalate if the peripheral "
            "remains incompatible."
        ),
        "elevated": (
            "Confirm the exact peripheral model, connection type and driver version. "
            "Provide the compatible driver or a supported alternative, and escalate to "
            "technical support if the customer depends on the device and it still does "
            "not work."
        ),
    },
    "Product compatibility": {
        "team": "Technical / Sales Support",
        "standard": (
            "Verify the customer's device, software version, and product requirements. "
            "Check compatibility specifications and recommend a supported configuration or "
            "alternative where necessary."
        ),
        "elevated": (
            "Verify the device, software version and product requirements immediately. If "
            "the product is confirmed incompatible, offer a supported alternative or the "
            "applicable exchange/return process and escalate if the customer is blocked."
        ),
    },
    "Product recommendation": {
        "team": "Sales Support",
        "standard": (
            "Identify the customer's requirements, intended use, and budget, then recommend "
            "the most suitable product based on those needs."
        ),
    },
    "Product setup": {
        "team": "Technical Support",
        "standard": (
            "Guide the customer through the required setup and configuration steps. Verify "
            "compatibility and required settings before escalating persistent setup "
            "failures."
        ),
        "elevated": (
            "Walk the customer through setup step by step, verify compatibility and "
            "firmware, and reset to factory defaults if needed. If the product still cannot "
            "be set up, escalate to technical support or arrange a replacement."
        ),
    },
    "Refund request": {
        "team": "Billing / Refunds Team",
        "standard": (
            "Verify the order and refund eligibility, check the current refund status, and "
            "initiate or track the refund according to the applicable refund process."
        ),
        "elevated": (
            "Verify the order and refund eligibility immediately and check the current "
            "refund status. If the refund is overdue or was wrongly declined, escalate to "
            "the billing/refunds team with the order details."
        ),
    },
    "Software bug": {
        "team": "Technical Support / Engineering",
        "standard": (
            "Reproduce the reported software issue, check for available updates, and review "
            "relevant error conditions. Apply the latest software fix or escalate the "
            "reproducible issue to the technical support team."
        ),
        "elevated": (
            "Reproduce the issue, collect error messages, logs and version details, and "
            "apply any known fix or workaround. If it blocks core functionality or affects "
            "customer data, escalate to engineering immediately."
        ),
    },
}


def _escalation_risk(escalation):
    return escalation.get("risk") if isinstance(escalation, dict) else None


def get_resolution_level(issue, priority, escalation):
    """Standard / Elevated / Critical - how strong the recommended response is."""
    entry = RESOLUTION_PLAYBOOK.get(issue, {})
    risk = _escalation_risk(escalation)

    if "critical" in entry and (priority == "Critical" or risk == "Critical"):
        return "Critical"
    if "elevated" in entry and (
        priority in ("Critical", "High") or risk in ("High", "Critical")
    ):
        return "Elevated"
    return "Standard"


def get_resolution(issue, priority, escalation):
    entry = RESOLUTION_PLAYBOOK.get(issue)

    if not entry:
        return resolution_map.get(issue, DEFAULT_RESOLUTION)

    level = get_resolution_level(issue, priority, escalation)
    return entry.get(level.lower(), entry["standard"])


def get_resolution_team(issue):
    return RESOLUTION_PLAYBOOK.get(issue, {}).get("team", "Appropriate support team")


# ============================================================
# ESCALATION RISK ENGINE
# ============================================================
# Rule tables live here once; the engine AND the website pages both read them,
# so what the page explains is always exactly what the engine does.

PRIORITY_POINTS = {"Critical": 4, "High": 3, "Medium": 1, "Low": 0}

ISSUE_POINTS = {
    "Data loss": 3,
    "Account access": 2,
    "Payment issue": 2,
    "Battery life": 2,
    "Hardware issue": 2,
    "Network problem": 2,
    "Software bug": 2,
    "Refund request": 1,
    "Delivery problem": 1,
    "Cancellation request": 1,
}

KEYWORD_SIGNALS = [
    {
        "icon": "🔐", "name": "Security", "points": 3,
        "reason": "Security or unauthorized-access language detected",
        "words": ["hacked", "compromised", "fraud", "unauthorized",
                  "security breach", "stolen account"],
    },
    {
        "icon": "💾", "name": "Data loss", "points": 3,
        "reason": "Potential data-loss risk detected",
        "words": ["lost data", "deleted data", "missing data",
                  "data loss", "files disappeared"],
    },
    {
        "icon": "💳", "name": "Financial impact", "points": 2,
        "reason": "Financial-impact language detected",
        "words": ["charged twice", "double charged", "duplicate charge",
                  "fraudulent charge", "money missing", "wrong amount"],
    },
    {
        "icon": "🛑", "name": "Service impact", "points": 2,
        "reason": "Severe service-impact language detected",
        "words": ["system down", "completely down", "everything stopped",
                  "not working at all"],
    },
    {
        "icon": "⏱️", "name": "Urgency", "points": 1,
        "reason": "Urgency language detected",
        "words": ["urgent", "immediately", "asap", "right now",
                  "emergency", "critical"],
    },
    {
        "icon": "⚖️", "name": "Legal", "points": 1,
        "reason": "Legal escalation language detected",
        "words": ["lawsuit", "legal action", "lawyer", "consumer court",
                  "complaint to authority"],
    },
]

# (minimum score, risk level, recommended action) - highest first
RISK_LEVELS = [
    (8, "Critical", "Immediate human escalation"),
    (6, "High", "Priority human escalation"),
    (3, "Medium", "Priority support queue"),
    (0, "Low", "Normal support workflow"),
]

RISK_RANGES = {"Low": "0–2", "Medium": "3–5", "High": "6–7", "Critical": "8+"}


def escalation_risk_engine(text, issue, priority):
    lowered = text.lower()
    breakdown = []  # [(label, points)]

    points = PRIORITY_POINTS.get(priority, 0)
    if points:
        breakdown.append((f"{priority} priority detected", points))

    points = ISSUE_POINTS.get(issue, 0)
    if points:
        breakdown.append((f"{issue} requires additional attention", points))

    for signal in KEYWORD_SIGNALS:
        if any(word in lowered for word in signal["words"]):
            breakdown.append((signal["reason"], signal["points"]))

    score = sum(points for _, points in breakdown)
    risk, action = next(
        (level, act) for minimum, level, act in RISK_LEVELS if score >= minimum
    )

    reasons = [f"{label} (+{points})" for label, points in breakdown]
    if not reasons:
        reasons.append("No strong escalation signals detected")

    return {
        "score": score,
        "risk": risk,
        "action": action,
        "reasons": reasons,
        "breakdown": breakdown,
    }


# ============================================================
# SHARED UI COMPONENTS (escalation + ML pages)
# ============================================================

METER_MAX = 12
METER_SEGMENTS = [("low", 3), ("medium", 3), ("high", 2), ("critical", 4)]

ESCALATION_SAMPLES = {
    "🔐 Hacked + urgent": "My account has been hacked and someone has made unauthorized transactions. This is urgent.",
    "💳 Double charge": "I was charged twice for the same order and I need this fixed immediately.",
    "⚖️ Legal threat": "All my files disappeared after the update. If this is not fixed I will take legal action.",
    "🙂 Simple query": "Can you recommend a good laptop for college within a small budget?",
}


def risk_meter_html(score):
    """Horizontal Low / Medium / High / Critical gauge with a marker at the score."""
    position = min(max(score, 0), METER_MAX) / METER_MAX * 100
    segments = "".join(
        f'<div class="meter-seg meter-{name}" style="flex:{width}"></div>'
        for name, width in METER_SEGMENTS
    )
    ticks = "".join(
        f'<span style="left:{value / METER_MAX * 100:.1f}%">{label}</span>'
        for value, label in [(0, "0"), (3, "3"), (6, "6"), (8, "8"), (12, "12+")]
    )
    return (
        '<div class="meter-wrap">'
        f'<div class="meter">{segments}<div class="meter-marker" style="left:{position:.1f}%"></div></div>'
        f'<div class="meter-scale">{ticks}</div>'
        "</div>"
    )


def breakdown_html(escalation):
    """One row per scoring signal, with a points bar."""
    rows = escalation.get("breakdown")

    if not rows:
        return "".join(
            f'<div class="reason-item">• {esc(reason)}</div>'
            for reason in escalation["reasons"]
        )

    return "".join(
        f"""
        <div class="bd-row">
            <div class="bd-label">{esc(label)}</div>
            <div class="bd-bar"><div class="bd-fill" style="width:{min(100, points / 4 * 100):.0f}%"></div></div>
            <div class="bd-pts">+{points}</div>
        </div>
        """
        for label, points in rows
    )


def escalation_card_html(escalation):
    risk_class = RISK_CLASS.get(escalation["risk"], "risk-low")
    return f"""
        <div class="escalation-card">
            <div class="res-head">
                <div class="resolution-title" style="margin-bottom:0;">🚨 Escalation Risk Engine</div>
                <span class="level-badge lvl-standard" style="color:#c7d4f0;border-color:rgba(160,180,230,0.3);background:rgba(90,120,200,0.10);">
                    Score {escalation["score"]}
                </span>
            </div>
            <div class="risk-hero">
                <span class="risk-score {risk_class}">{esc(escalation["risk"])}</span>
                <span class="risk-action">→ {esc(escalation["action"])}</span>
            </div>
            {risk_meter_html(escalation["score"])}
            <div class="section-label" style="margin-top:6px;">Score breakdown</div>
            {breakdown_html(escalation)}
        </div>
    """


def signal_rules_html():
    """The scoring rules, generated from the same tables the engine uses."""
    priority_pills = "".join(
        f'<span class="rule-pill"><b>{esc(name)}</b> +{points}</span>'
        for name, points in PRIORITY_POINTS.items()
    )

    by_points = {}
    for issue, points in ISSUE_POINTS.items():
        by_points.setdefault(points, []).append(issue)

    issue_rows = "".join(
        f'<div class="rule-row"><span class="rule-pts">+{points}</span>'
        f'<span class="rule-text">{esc(", ".join(sorted(issues)))}</span></div>'
        for points, issues in sorted(by_points.items(), reverse=True)
    )

    signal_rows = "".join(
        f'<div class="rule-row"><span class="rule-pts">+{s["points"]}</span>'
        f'<span class="rule-text"><b>{s["icon"]} {esc(s["name"])}</b> · '
        f'{esc(", ".join(s["words"][:4]))}…</span></div>'
        for s in KEYWORD_SIGNALS
    )

    return f"""
        <div class="glass-card">
            <div class="card-title">How the score is calculated</div>

            <div class="section-label" style="margin-top:6px;">1 · Priority</div>
            <div class="rule-pills">{priority_pills}</div>

            <div class="section-label" style="margin-top:16px;">2 · Issue type</div>
            {issue_rows}

            <div class="section-label" style="margin-top:16px;">3 · Risk language in the complaint</div>
            {signal_rows}

            <div class="section-label" style="margin-top:16px;">4 · Total → risk level</div>
            <div class="rule-pills">
                <span class="rule-pill risk-low">Low 0–2</span>
                <span class="rule-pill risk-medium">Medium 3–5</span>
                <span class="rule-pill risk-high">High 6–7</span>
                <span class="rule-pill risk-critical">Critical 8+</span>
            </div>
        </div>
    """


def kv_card(title, rows):
    body = "".join(
        f'<div class="kv-row"><span class="kv-key">{esc(key)}</span>'
        f'<span class="kv-val">{esc(value)}</span></div>'
        for key, value in rows
    )
    return f'<div class="glass-card"><div class="card-title">{esc(title)}</div>{body}</div>'


# ---------- ML Intelligence content ----------

PIPELINE_STAGES = [
    {
        "icon": "🗂️", "title": "Dataset", "short": "8,469 tickets · 17 features",
        "flow": ["Raw tickets", "Data cleaning", "Labelled dataset"],
        "body": (
            "The system is built on 8,469 customer support tickets with 17 features, "
            "labelled with 16 issue categories and 4 priority levels. This is the data "
            "foundation for the issue prototypes and the priority reference examples."
        ),
    },
    {
        "icon": "🧹", "title": "Text Processing", "short": "Subject + description",
        "flow": ["Ticket subject", "+ Description", "Text cleaning", "Ticket text"],
        "body": (
            "Raw tickets are cleaned, and the ticket subject and description are combined "
            "into one text field. That combined text is the input for the NLP model."
        ),
    },
    {
        "icon": "🧬", "title": "Sentence Embeddings", "short": "MiniLM · 384 dimensions",
        "flow": ["Customer complaint", "Sentence Transformer", "384-dim embedding"],
        "body": (
            "The all-MiniLM-L6-v2 sentence transformer turns text into a 384-dimensional "
            "vector that captures meaning. Complaints that mean the same thing, such as "
            "“my order hasn't arrived” and “the package is late”, end up close together "
            "even when the words differ."
        ),
    },
    {
        "icon": "⚡", "title": "Issue Detection", "short": "Prototype similarity",
        "flow": ["Embedding", "Compare with 16 prototypes", "Cosine similarity",
                 "Highest match", "Predicted issue"],
        "body": (
            "Each of the 16 issue categories has a prototype embedding built from the "
            "training tickets. A new complaint is compared with every prototype using "
            "cosine similarity, and the closest one is the predicted issue. The top three "
            "matches are shown to the support agent."
        ),
    },
    {
        "icon": "🎯", "title": "Priority", "short": "Semantic + safety override",
        "flow": ["Complaint", "Semantic priority analysis", "Critical keyword check", "Priority"],
        "body": (
            "The complaint is compared with reference examples of Critical, High, Medium "
            "and Low situations. A critical-keyword check (hacked, fraud, unauthorized and "
            "similar) then overrides the result to Critical for serious security cases."
        ),
    },
    {
        "icon": "🚨", "title": "Escalation Risk", "short": "Transparent rule-based score",
        "flow": ["Issue + Priority", "Risk signals", "Risk score", "Risk level", "Recommended action"],
        "body": (
            "A rule-based engine adds points for priority, issue type and risk language "
            "(security, data loss, financial impact, service outage, urgency, legal). The "
            "total maps to Low, Medium, High or Critical, and every point is listed as a "
            "reason. It is rule-based on purpose, so support teams can see exactly why a "
            "case was escalated."
        ),
    },
    {
        "icon": "💡", "title": "Resolution", "short": "Issue × severity playbook",
        "flow": ["Issue", "Priority", "Escalation", "Recommended resolution"],
        "body": (
            "The recommended action depends on the issue, the priority and the escalation "
            "result. Each of the 16 issues has a standard and an elevated response, "
            "sensitive ones also have a critical response, and every case is routed to the "
            "right team."
        ),
    },
    {
        "icon": "💬", "title": "AI Assistant", "short": "Local Llama 3.2 3B",
        "flow": ["Analysis results", "Local LLM (Ollama)", "Plain-language explanation"],
        "body": (
            "A local Llama 3.2 3B model, run through Ollama, explains the results in plain "
            "language. It receives the latest analysis, so agents can ask why a complaint is "
            "Critical or what to do next. Nothing leaves the machine."
        ),
    },
]


def stage_grid_html():
    cards = "".join(
        f"""
        <div class="stage-card">
            <span class="stage-num">{number:02d}</span>
            <div class="stage-icon">{stage["icon"]}</div>
            <div class="stage-title">{esc(stage["title"])}</div>
            <div class="stage-short">{esc(stage["short"])}</div>
        </div>
        """
        for number, stage in enumerate(PIPELINE_STAGES, 1)
    )
    return f'<div class="stage-grid">{cards}</div>'


def stage_extra_html(title):
    """Extra, live content for selected pipeline stages."""
    if title == "Issue Detection":
        tags = "".join(f'<span class="tag">{esc(name)}</span>' for name in sorted(issue_prototypes))
        return (
            f'<div class="section-label" style="margin-top:18px;">{len(issue_prototypes)} issue categories</div>'
            f"{tags}"
        )

    if title == "Priority":
        tags = "".join(
            f'<span class="tag {PRIORITY_CLASS[name]}">{name}</span>'
            for name in ("Critical", "High", "Medium", "Low")
        )
        return f'<div class="section-label" style="margin-top:18px;">Priority levels</div>{tags}'

    if title == "Escalation Risk":
        tags = "".join(
            f'<span class="tag {RISK_CLASS[level]}">{level} · score {RISK_RANGES[level]}</span>'
            for _, level, _ in reversed(RISK_LEVELS)
        )
        return f'<div class="section-label" style="margin-top:18px;">Risk levels</div>{tags}'

    return ""


# ============================================================
# ANALYZE COMPLAINT
# ============================================================

def analyze_complaint(text):

    issue_scores = get_issue_scores(text)
    issue = issue_scores[0][0]

    priority = predict_priority(text)
    escalation = escalation_risk_engine(text, issue, priority)
    resolution = get_resolution(issue, priority, escalation)
    resolution_level = get_resolution_level(issue, priority, escalation)
    resolution_team = get_resolution_team(issue)

    result = {
        "text": text,
        "issue": issue,
        "priority": priority,
        "resolution": resolution,
        "resolution_level": resolution_level,
        "resolution_team": resolution_team,
        "escalation": escalation,
        "top_matches": issue_scores[:3],
        "timestamp": datetime.now(),
    }
    result["pdf"] = build_pdf_report(result)  # built once, downloaded instantly
    return result


def build_report(result):
    esc_data = result["escalation"]
    lines = [
        "SUPPORTAI - COMPLAINT ANALYSIS REPORT",
        "=" * 44,
        f"Generated   : {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "CUSTOMER COMPLAINT",
        result["text"],
        "",
        f"Detected Issue  : {result['issue']}",
        f"Priority        : {result['priority']}",
        f"Escalation Risk : {esc_data['risk']} (score {esc_data['score']})",
        f"Recommended Path: {esc_data['action']}",
        f"Route To        : {result['resolution_team']}",
        "",
        "RISK SIGNALS",
        *[f"  - {r}" for r in esc_data["reasons"]],
        "",
        "RECOMMENDED RESOLUTION",
        result["resolution"],
    ]
    return "\n".join(lines)


# ============================================================
# PDF REPORT  (pure Python - no extra library to install)
# ============================================================

# Helvetica character widths (1/1000 em) for ASCII 32..126 - used for text wrapping
_HELV_WIDTHS = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556,
    278, 278, 584, 584, 584, 556, 1015,
    667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833,
    722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611,
    278, 278, 278, 469, 556, 333,
    556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833,
    556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500,
    334, 260, 334, 584,
]
assert len(_HELV_WIDTHS) == 95

PDF_NAVY = (0.04, 0.07, 0.15)
PDF_BLUE = (0.23, 0.45, 0.95)
PDF_WHITE = (1, 1, 1)
PDF_DARK = (0.11, 0.14, 0.22)
PDF_GREY = (0.42, 0.47, 0.58)
PDF_LIGHT = (0.95, 0.96, 0.985)
PDF_TRACK = (0.88, 0.90, 0.94)
PDF_PRIORITY = {
    "Critical": (0.85, 0.21, 0.25),
    "High": (0.88, 0.45, 0.10),
    "Medium": (0.78, 0.58, 0.04),
    "Low": (0.12, 0.62, 0.38),
}


def _pdf_clean(text):
    text = str(text).replace("→", "->").replace("…", "...").replace("\t", " ").replace("\r", "")
    return text.encode("cp1252", "replace").decode("cp1252")


def _pdf_escape(text):
    return _pdf_clean(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class _PdfDoc:
    W, H, M = 595.28, 841.89, 42.0  # A4 in points, page margin

    def __init__(self):
        self.pages = []
        self.new_page()

    def new_page(self):
        self.ops = []
        self.pages.append(self.ops)
        self.y = self.H - self.M

    def need(self, height):
        """Start a new page when `height` doesn't fit above the footer."""
        if self.y - height < self.M + 30:
            self.new_page()

    @staticmethod
    def width(text, size, bold=False):
        total = sum(
            _HELV_WIDTHS[ord(c) - 32] if 32 <= ord(c) <= 126 else 556
            for c in _pdf_clean(text)
        )
        return total * size / 1000 * (1.07 if bold else 1.0)

    def wrap(self, text, size, max_width, bold=False):
        lines = []
        for paragraph in _pdf_clean(text).split("\n"):
            line = ""
            for word in paragraph.split():
                while self.width(word, size, bold) > max_width:  # very long word
                    cut = len(word)
                    while cut > 1 and self.width(word[:cut], size, bold) > max_width:
                        cut -= 1
                    if line:
                        lines.append(line)
                        line = ""
                    lines.append(word[:cut])
                    word = word[cut:]
                trial = f"{line} {word}".strip()
                if self.width(trial, size, bold) <= max_width:
                    line = trial
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        return lines or [""]

    def rect(self, x, y, w, h, color):
        self.ops.append("%.3f %.3f %.3f rg %.2f %.2f %.2f %.2f re f" % (*color, x, y, w, h))

    def text(self, x, y, s, size=10, bold=False, color=PDF_DARK):
        self.ops.append(
            "BT /%s %.1f Tf %.3f %.3f %.3f rg %.2f %.2f Td (%s) Tj ET"
            % ("F2" if bold else "F1", size, *color, x, y, _pdf_escape(s))
        )

    def text_right(self, x_right, y, s, size=10, bold=False, color=PDF_DARK):
        self.text(x_right - self.width(s, size, bold), y, s, size, bold, color)

    def to_bytes(self, title):
        count = len(self.pages)
        info_id = 5 + 2 * count
        out = bytearray(b"%PDF-1.4\n")
        offsets = {}

        def add(num, body):
            offsets[num] = len(out)
            out.extend(f"{num} 0 obj\n".encode() + body + b"\nendobj\n")

        kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(count))
        add(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        add(2, f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode())
        for num, name in ((3, "Helvetica"), (4, "Helvetica-Bold")):
            add(num, f"<< /Type /Font /Subtype /Type1 /BaseFont /{name} /Encoding /WinAnsiEncoding >>".encode())

        for i, ops in enumerate(self.pages):
            stream = "\n".join(ops).encode("cp1252", "replace")
            add(
                5 + 2 * i,
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.W} {self.H}] "
                    f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {6 + 2 * i} 0 R >>"
                ).encode(),
            )
            add(6 + 2 * i, f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")

        add(info_id, f"<< /Title ({_pdf_escape(title)}) /Producer (SupportAI) >>".encode())

        xref_at = len(out)
        total = info_id + 1
        out.extend(f"xref\n0 {total}\n0000000000 65535 f \n".encode())
        for num in range(1, total):
            out.extend(f"{offsets[num]:010d} 00000 n \n".encode())
        out.extend(
            f"trailer\n<< /Size {total} /Root 1 0 R /Info {info_id} 0 R >>\nstartxref\n{xref_at}\n%%EOF".encode()
        )
        return bytes(out)


def build_pdf_report(result):
    """One- or two-page professional analysis report. Returns PDF bytes."""
    d = _PdfDoc()
    W, H, M = d.W, d.H, d.M
    cw = W - 2 * M
    escalation = result["escalation"]
    priority, risk, score = result["priority"], escalation["risk"], escalation["score"]
    risk_color = PDF_PRIORITY.get(risk, PDF_BLUE)
    stamp = result["timestamp"]

    # ---- header band ----
    d.rect(0, H - 92, W, 92, PDF_NAVY)
    d.rect(0, H - 95, W, 3, PDF_BLUE)
    d.text(M, H - 44, "SupportAI", 24, True, PDF_WHITE)
    d.text(M, H - 64, "Customer Support Intelligence  |  Complaint Analysis Report", 10, False, (0.72, 0.78, 0.92))
    d.text_right(W - M, H - 44, stamp.strftime("%d %b %Y"), 11, True, PDF_WHITE)
    d.text_right(W - M, H - 62, stamp.strftime("%H:%M:%S"), 10, False, (0.72, 0.78, 0.92))
    d.y = H - 124

    def label(text, room=70):
        d.need(room)
        d.text(M, d.y, text.upper(), 8, True, PDF_GREY)
        d.y -= 14

    # ---- complaint ----
    label("Customer complaint")
    lines = d.wrap(result["text"][:1500], 10.5, cw - 28)
    h = len(lines) * 14.5 + 20
    d.need(h)
    d.rect(M, d.y - h, cw, h, PDF_LIGHT)
    d.rect(M, d.y - h, 3, h, PDF_BLUE)
    ty = d.y - 18
    for line in lines:
        d.text(M + 14, ty, line, 10.5)
        ty -= 14.5
    d.y -= h + 20

    # ---- summary cards ----
    label("Analysis summary")
    cards = [
        ("DETECTED ISSUE", result["issue"], PDF_BLUE),
        ("PRIORITY", priority, PDF_PRIORITY.get(priority, PDF_BLUE)),
        ("ESCALATION RISK", f"{risk} ({score})", risk_color),
        ("NEXT ACTION", escalation["action"], risk_color),
    ]
    gap, card_h = 10, 76
    card_w = (cw - 3 * gap) / 4
    d.need(card_h)
    for i, (title, value, color) in enumerate(cards):
        x = M + i * (card_w + gap)
        d.rect(x, d.y - card_h, card_w, card_h, PDF_LIGHT)
        d.rect(x, d.y - card_h, card_w, 3, color)
        d.text(x + 10, d.y - 20, title, 7, True, PDF_GREY)
        for j, line in enumerate(d.wrap(value, 11.5, card_w - 20, True)[:3]):
            d.text(x + 10, d.y - 39 - j * 13, line, 11.5, True, color)
    d.y -= card_h + 22

    # ---- resolution ----
    label("Recommended resolution")
    lines = d.wrap(result["resolution"], 10.5, cw - 28)
    h = len(lines) * 14.5 + 70
    d.need(h)
    d.rect(M, d.y - h, cw, h, PDF_LIGHT)
    d.rect(M, d.y - h, 3, h, PDF_BLUE)
    d.text(M + 14, d.y - 19, f"{result['resolution_level']} response   |   Route to: {result['resolution_team']}", 9, True, PDF_BLUE)
    ty = d.y - 40
    for line in lines:
        d.text(M + 14, ty, line, 10.5)
        ty -= 14.5
    d.rect(M + 14, ty - 2, cw - 28, 0.6, PDF_TRACK)
    d.text(M + 14, ty - 18, "Recommended action:", 10, True)
    d.text(M + 14 + d.width("Recommended action:  ", 10, True), ty - 18, escalation["action"], 10, True, risk_color)
    d.y -= h + 22

    # ---- escalation breakdown ----
    label("Escalation risk breakdown", room=150)
    segments = [(3, PDF_PRIORITY["Low"]), (3, PDF_PRIORITY["Medium"]), (2, PDF_PRIORITY["High"]), (4, PDF_PRIORITY["Critical"])]
    x = M
    for weight, color in segments:
        seg_w = cw * weight / 12
        d.rect(x, d.y - 9, seg_w - 2, 8, color)
        x += seg_w
    d.rect(M + cw * min(score, 12) / 12 - 1.5, d.y - 13, 3, 16, PDF_DARK)
    for value, text_ in ((0, "0"), (3, "3"), (6, "6"), (8, "8")):
        d.text(M + cw * value / 12, d.y - 25, text_, 7, False, PDF_GREY)
    d.text_right(M + cw, d.y - 25, "12+", 7, False, PDF_GREY)
    d.y -= 46

    rows = escalation.get("breakdown") or []
    if not rows:
        d.text(M, d.y, "No strong escalation signals detected.", 10, False, PDF_GREY)
        d.y -= 20
    for text_, points in rows:
        d.need(24)
        d.text(M, d.y, text_, 10)
        d.rect(M + 330, d.y - 1, 110, 5, PDF_TRACK)
        d.rect(M + 330, d.y - 1, 110 * min(1.0, points / 4), 5, risk_color)
        d.text_right(M + cw, d.y, f"+{points}", 10, True)
        d.y -= 19
    d.need(30)
    d.rect(M, d.y + 8, cw, 0.6, PDF_TRACK)
    d.y -= 8
    d.text(M, d.y, "Total risk score", 10.5, True)
    d.text_right(M + cw, d.y, f"{score} / 12   -   {risk}", 11, True, risk_color)
    d.y -= 30

    # ---- closest categories ----
    label("Closest issue categories", room=72)
    for name, similarity in result["top_matches"]:
        d.need(24)
        pct = max(0, min(100, round(similarity * 100)))
        d.text(M, d.y, name, 10)
        d.rect(M + 250, d.y - 1, 190, 5, PDF_TRACK)
        d.rect(M + 250, d.y - 1, 190 * pct / 100, 5, PDF_BLUE)
        d.text_right(M + cw, d.y, f"{pct}%", 10, True)
        d.y -= 19

    # ---- footer on every page ----
    total_pages = len(d.pages)
    for number, ops in enumerate(d.pages, 1):
        d.ops = ops
        d.rect(M, 46, cw, 0.6, PDF_TRACK)
        d.text(M, 32, "Generated by SupportAI. AI-assisted recommendation: a human agent should verify before acting.", 7.5, False, PDF_GREY)
        d.text_right(W - M, 32, f"Page {number} of {total_pages}", 7.5, False, PDF_GREY)

    return d.to_bytes("SupportAI Complaint Analysis Report")


# ============================================================
# DASHBOARD LOG  (saved to disk so the KPIs survive restarts)
# ============================================================

LOG_FILE = Path(__file__).resolve().parent / "analysis_log.jsonl"
KPI_SCORE_MAX = 12


def log_analysis(result):
    entry = {
        "ts": result["timestamp"].isoformat(timespec="seconds"),
        "issue": result["issue"],
        "priority": result["priority"],
        "risk": result["escalation"]["risk"],
        "score": result["escalation"]["score"],
    }
    try:
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # logging must never break the app


def read_log(limit=5000):
    try:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []

    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
            entry["dt"] = datetime.fromisoformat(entry["ts"])
            entries.append(entry)
        except (ValueError, KeyError, TypeError):
            continue
    return entries


def reset_stats():
    try:
        LOG_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    st.session_state.history = []


def compute_kpis(entries):
    total = len(entries)

    if not total:
        muted = "sub-muted"
        return [
            ("📥", "Complaints analyzed", "0", "Run your first analysis", muted),
            ("🔥", "Critical cases", "0", "No data yet", muted),
            ("🚨", "Human escalations", "0", "No data yet", muted),
            ("📈", "Avg. risk score", f"– / {KPI_SCORE_MAX}", "No data yet", muted),
        ]

    today = datetime.now().date()
    n_today = sum(1 for e in entries if e["dt"].date() == today)
    n_yesterday = sum(1 for e in entries if e["dt"].date() == today - timedelta(days=1))

    if n_yesterday:
        change = (n_today - n_yesterday) / n_yesterday * 100
        trend = f"{'↑' if change >= 0 else '↓'} {abs(change):.0f}% vs yesterday"
        trend_class = "sub-up" if change >= 0 else "sub-down"
    else:
        trend = f"{n_today} today"
        trend_class = "sub-up" if n_today else "sub-muted"

    critical = sum(1 for e in entries if e["priority"] == "Critical")
    escalated = sum(1 for e in entries if e["risk"] in ("High", "Critical"))
    average = sum(e["score"] for e in entries) / total
    average_level = next(level for minimum, level, _ in RISK_LEVELS if average >= minimum)

    return [
        ("📥", "Complaints analyzed", f"{total:,}", trend, trend_class),
        ("🔥", "Critical cases", str(critical), f"{critical / total * 100:.1f}% of complaints",
         "risk-critical" if critical else "sub-muted"),
        ("🚨", "Human escalations", str(escalated), f"{escalated / total * 100:.1f}% escalation rate",
         "risk-high" if escalated else "sub-muted"),
        ("📈", "Avg. risk score", f"{average:.1f} / {KPI_SCORE_MAX}", average_level,
         RISK_CLASS.get(average_level, "sub-muted")),
    ]


def render_kpi_row():
    for column, (icon, label, value, sub, sub_class) in zip(
        st.columns(4, gap="medium"), compute_kpis(read_log())
    ):
        with column:
            render_html(
                f"""
                <div class="kpi">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-body">
                        <div class="kpi-label">{esc(label)}</div>
                        <div class="kpi-value">{esc(value)}</div>
                        <div class="kpi-sub {sub_class}">{esc(sub)}</div>
                    </div>
                </div>
                """
            )


def insights_html(entries):
    """Priority mix + most common issues, built from the saved log."""
    if not entries:
        return None

    colors = {"Critical": "#ff6574", "High": "#ff9f5b", "Medium": "#f4c95d", "Low": "#5cdd9a"}
    counts = Counter(e["priority"] for e in entries)

    bar = "".join(
        f'<div style="flex:{counts[p]};background:{colors[p]}"></div>' for p in colors if counts[p]
    )
    legend = "".join(
        f'<span class="legend-item"><i style="background:{colors[p]}"></i>{p} <b>{counts[p]}</b></span>'
        for p in colors if counts[p]
    )

    top = Counter(e["issue"] for e in entries).most_common(3)
    peak = top[0][1]
    issue_rows = "".join(
        f"""
        <div class="match-row">
            <div class="match-head"><span>{esc(name)}</span><span>{count}</span></div>
            <div class="match-bar"><div class="match-fill" style="width:{count / peak * 100:.0f}%"></div></div>
        </div>
        """
        for name, count in top
    )

    priority_card = f"""
        <div class="glass-card">
            <div class="card-title">Priority mix</div>
            <div class="stack-bar">{bar}</div>
            <div class="legend">{legend}</div>
        </div>
    """
    issues_card = f"""
        <div class="glass-card">
            <div class="card-title">Top issues</div>
            {issue_rows}
        </div>
    """
    return priority_card, issues_card


# ============================================================
# AI ASSISTANT (shared by both chat panels)
# ============================================================

# Short answers + low temperature = fast and consistent.
# keep_alive keeps the model loaded in RAM so replies after the first are fast.
CHAT_OPTIONS = {"num_predict": 120, "temperature": 0.2, "num_ctx": 3072}
CHAT_KEEP_ALIVE = "30m"

OLLAMA_UNAVAILABLE = (
    "I couldn't reach the local AI assistant. Start the Ollama app (or run "
    "`ollama serve` in a terminal) and make sure the model is installed with "
    "`ollama pull llama3.2:3b`."
)

OLLAMA_PACKAGE_MISSING = (
    "The `ollama` Python package is missing from the Python that is running this "
    f"app (`{sys.executable}`). Install it into that exact Python by running: "
    f"`\"{sys.executable}\" -m pip install ollama` then restart the app."
)

OLLAMA_MODEL_MISSING = (
    "The model isn't installed yet. Run `ollama pull llama3.2:3b` in a terminal, "
    "then try again."
)


def llm_error_message(exc):
    """Turn an Ollama failure into a specific, actionable message."""
    text = str(exc).lower()
    if ollama is None or isinstance(exc, ImportError):
        return OLLAMA_PACKAGE_MISSING
    if getattr(exc, "status_code", None) == 404 or ("model" in text and "not found" in text):
        return OLLAMA_MODEL_MISSING
    return OLLAMA_UNAVAILABLE


def llm_status_hint():
    """None when the assistant is ready, otherwise what the user should do."""
    if ollama is None:
        return OLLAMA_PACKAGE_MISSING
    if not ollama_online():
        return OLLAMA_UNAVAILABLE
    return None

CREDENTIAL_WARNING = (
    "For security, please don't share passwords, OTPs, payment PINs or card details "
    "in this chat. For a compromised account, start the official account-recovery "
    "process and escalate the case to the security team."
)

# What the chatbot knows about THIS project (so it can explain the system)
PROJECT_KNOWLEDGE = """
About the SupportAI system:
- Semantic Issue Detection: the complaint is converted into a sentence embedding (all-MiniLM-L6-v2) and compared by cosine similarity with a prototype embedding for each of 16 issue categories. The closest category is the detected issue.
- Priority Prediction: the complaint is compared with reference examples of Critical, High, Medium and Low situations. Security words such as hacked, fraud or unauthorized force Critical.
- Escalation Risk Engine: a transparent rule-based score. Points: priority (Critical 4, High 3, Medium 1, Low 0); issue type (Data loss 3; Account access, Payment issue, Battery life, Hardware issue, Network problem, Software bug 2; Refund, Delivery, Cancellation 1); security language +3; data-loss language +3; financial-impact language +2; severe service impact +2; urgency +1; legal language +1. Total 8 or more = Critical, 6-7 = High, 3-5 = Medium, below 3 = Low.
- Resolution: a recommended action is chosen for the detected issue and priority.
- Pipeline: Complaint -> Semantic issue -> Priority -> Escalation risk -> Resolution. This assistant explains those results to the support agent.
""".strip()

HIGH_RISK_ISSUES = {"Account access", "Payment issue", "Data loss"}


def analysis_context_text():
    """Everything the bot needs to know about the latest analysed complaint."""
    analysis = st.session_state.analysis_result

    if not analysis:
        return (
            "No complaint has been analyzed yet. If the user asks about a specific "
            "complaint, ask them to analyze one first. You can still explain how "
            "the system works."
        )

    esc_data = analysis["escalation"]
    matches = ", ".join(f"{name} ({sim * 100:.0f}%)" for name, sim in analysis["top_matches"])
    reasons = "; ".join(esc_data["reasons"])

    context = f"""
LATEST ANALYZED COMPLAINT
Complaint text: {analysis["text"][:500]}
Detected issue: {analysis["issue"]} (closest matches: {matches})
Priority: {analysis["priority"]}
Escalation risk: {esc_data["risk"]} (score {esc_data["score"]})
Why (escalation reasons): {reasons}
Recommended handling: {esc_data["action"]}
Route to team: {analysis["resolution_team"]}
Recommended resolution ({analysis["resolution_level"]} level): {analysis["resolution"]}
""".strip()

    if (
        analysis["priority"] in ("Critical", "High")
        or esc_data["risk"] in ("High", "Critical")
        or analysis["issue"] in HIGH_RISK_ISSUES
    ):
        context += (
            "\nNOTE: This is a sensitive/high-risk case. Recommend human escalation "
            "where appropriate and never ask the customer for credentials."
        )

    return context


KNOWLEDGE_TRIGGERS = (
    "semantic", "embedding", "prototype", "cosine", "priority prediction",
    "escalation", "score", "risk", "threshold", "pipeline", "how does",
    "how do", "how is", "how are", "system", "model", "work", "points",
)


def needs_project_knowledge(text):
    """Only send the (long) system description when the question is about the system."""
    lowered = (text or "").lower()
    return any(trigger in lowered for trigger in KNOWLEDGE_TRIGGERS)


SYSTEM_ANSWERS = {
    "What is semantic issue detection?": (
        "Semantic issue detection converts the complaint into a 384-dimensional sentence "
        "embedding (all-MiniLM-L6-v2) and compares it with a prototype embedding for each "
        "of the 16 issue categories using cosine similarity. The closest prototype is the "
        "detected issue, so complaints are matched by meaning rather than exact keywords."
    ),
    "What does Priority Prediction do?": (
        "Priority Prediction compares the complaint with reference examples of Critical, "
        "High, Medium and Low situations and picks the closest match. A safety override "
        "forces Critical when serious security wording such as hacked, fraud or "
        "unauthorized appears."
    ),
}


def instant_answer(question):
    """Answers built directly from the analysis: instant, accurate, no LLM needed."""
    if question in SYSTEM_ANSWERS:
        return SYSTEM_ANSWERS[question]

    analysis = st.session_state.analysis_result
    if not analysis:
        return None

    esc_data = analysis["escalation"]
    issue, priority = analysis["issue"], analysis["priority"]
    team = analysis["resolution_team"]
    drivers = "; ".join(f"{label} (+{pts})" for label, pts in esc_data["breakdown"]) \
        or "no strong escalation signals"

    if question == "Explain this complaint.":
        return (
            f"This complaint is classified as a **{issue}** with **{priority}** priority. "
            f"The escalation risk is **{esc_data['risk']}** (score {esc_data['score']}), "
            f"based on: {drivers}. Recommended action: **{esc_data['action']}**, "
            f"routed to the **{team}**."
        )

    if question.startswith("Why was this complaint classified as"):
        lowered = analysis["text"].lower()
        hits = [word for word in CRITICAL_KEYWORDS if word in lowered]
        if priority == "Critical" and hits:
            reason = f"it contains security-sensitive wording (\u201c{hits[0]}\u201d), which triggers the Critical safety override"
        else:
            reason = f"its meaning is closest to the reference examples for {priority} priority"
        return (
            f"The complaint is **{priority}** priority because {reason}. "
            f"This adds **+{PRIORITY_POINTS.get(priority, 0)}** to the escalation score."
        )

    if question == "What should the support agent do next?":
        answer = (
            f"Route this to the **{team}**. {analysis['resolution']} "
            f"Recommended handling: **{esc_data['action']}**."
        )
        if issue in HIGH_RISK_ISSUES:
            answer += " Never ask the customer for passwords, OTPs or PINs."
        return answer

    if question == "Explain the escalation risk for this complaint.":
        return (
            f"The escalation score is **{esc_data['score']}**, which gives **{esc_data['risk']}** "
            f"risk and the action **{esc_data['action']}**. Breakdown: {drivers}. "
            "Levels: Low 0-2, Medium 3-5, High 6-7, Critical 8+."
        )

    return None


def build_system_prompt(question=""):
    knowledge = f"\n{PROJECT_KNOWLEDGE}\n" if (needs_project_knowledge(question) or not st.session_state.analysis_result) else ""
    return f"""You are SupportAI, an AI customer support assistant for support agents.

Help users understand customer complaints, troubleshooting, priority, escalation risk and recommended resolutions.

Rules:
- Be concise and practical: 2 to 5 sentences. No long essays, no long lists.
- Never ask for passwords, OTPs, payment PINs, card numbers or any sensitive credentials. Tell customers not to share them.
- If a complaint involves security, fraud, data loss or serious account compromise, recommend human escalation.
- Do not invent company policies, numbers or facts. Use only the analysis and system description below.
- When asked "why", explain using the escalation reasons and score from the analysis.
- When asked to explain a complaint, mention the issue, priority, escalation risk, the main reason, and the recommended action.
{knowledge}
{analysis_context_text()}
"""


# ---------- safety helpers ----------

CREDENTIAL_SHARED = re.compile(
    r"\b(password|passcode|otp|pin|cvv)\b\s*(?:is|=|:)\s*\S+", re.IGNORECASE
)
CREDENTIAL_WORDS = re.compile(
    r"\b(password|passcode|otp|one[- ]time (?:password|code)|pin|cvv|card number)\b",
    re.IGNORECASE,
)
ASKING_VERBS = re.compile(
    r"\b(enter|share|provide|send|give|tell|type|confirm|verify|what is|what's)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(
    r"\b(do not|don't|dont|never|avoid|not|without|won't)\b", re.IGNORECASE
)


def contains_credentials(text):
    """True if the agent pasted an actual password / OTP / PIN into the chat."""
    return bool(CREDENTIAL_SHARED.search(text))


def sanitize_response(text):
    """Drop any sentence that asks for credentials and trim cut-off sentences."""
    text = (text or "").strip()
    if not text:
        return OLLAMA_UNAVAILABLE

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept, removed = [], False

    for sentence in sentences:
        if (
            CREDENTIAL_WORDS.search(sentence)
            and ASKING_VERBS.search(sentence)
            and not NEGATION.search(sentence)
        ):
            removed = True
            continue
        kept.append(sentence)

    # response hit the token limit mid-sentence -> drop the fragment
    if len(kept) > 1 and kept[-1][-1] not in ".!?)\"'`*":
        kept = kept[:-1]

    result = " ".join(kept).strip()

    if removed:
        result += " For security, never share passwords, OTPs or payment PINs in this chat."

    return result or CREDENTIAL_WARNING


# ---------- LLM call ----------

def stream_llm(messages):
    if ollama is None:
        raise ImportError("ollama package not installed")
    stream = ollama.Client(timeout=180).chat(
        model=LLM_MODEL,
        messages=messages,
        options=CHAT_OPTIONS,
        keep_alive=CHAT_KEEP_ALIVE,
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


@st.cache_resource
def warm_up_llm():
    """Load the model into RAM in the background so the first answer is quicker."""

    def _warm():
        try:
            # SAME options as real chats - different options would force a model reload
            ollama.Client(timeout=180).generate(
                model=LLM_MODEL,
                prompt="hi",
                options={**CHAT_OPTIONS, "num_predict": 1},
                keep_alive=CHAT_KEEP_ALIVE,
            )
        except Exception:
            pass

    threading.Thread(target=_warm, daemon=True).start()
    return True


# ---------- quick questions ----------

def queue_user_message(text):
    st.session_state.chat_messages.append({"role": "user", "content": text})


def get_quick_questions(include_system=False):
    analysis = st.session_state.analysis_result
    priority = analysis["priority"] if analysis else None

    why_label = f"⚠️ Why is it {priority}?" if priority else "⚠️ Why this priority?"
    why_question = (
        f"Why was this complaint classified as {priority} priority?"
        if priority
        else "Why was this complaint given this priority?"
    )

    questions = [
        ("📝 Explain this complaint", "Explain this complaint.", True),
        (why_label, why_question, True),
        ("🛠️ What should I do?", "What should the support agent do next?", True),
        ("🚨 Explain escalation risk", "Explain the escalation risk for this complaint.", True),
    ]

    if include_system:
        questions += [
            ("⚡ Semantic detection", "What is semantic issue detection?", False),
            ("🎯 Priority prediction", "What does Priority Prediction do?", False),
        ]

    return questions


# ---------- chat panel ----------

USE_FRAGMENTS = True  # set to False if the chat ever misbehaves on your Streamlit version
_chat_fragment = st.fragment if (USE_FRAGMENTS and hasattr(st, "fragment")) else (lambda fn: fn)


def rerun_chat():
    """Re-run only the chat panel (not the whole page) when fragments are on."""
    if USE_FRAGMENTS and hasattr(st, "fragment"):
        st.rerun(scope="fragment")
    st.rerun()


@_chat_fragment
def render_chat(key, include_system_questions=False, quick_cols=2):
    """Welcome bubble + history + streamed reply + quick questions + input."""

    messages = st.session_state.chat_messages
    analysis = st.session_state.analysis_result

    hint = llm_status_hint()
    if hint:
        st.warning(hint, icon="⚠️")

    # Welcome message (not stored in history)
    if not messages:
        with st.chat_message("assistant", avatar="🤖"):
            if analysis:
                st.markdown(
                    f"Hello! I've loaded your latest analysis "
                    f"(**{analysis['issue']}**, **{analysis['priority']}** priority). "
                    "Ask me to explain it, why it was classified this way, or what to do next."
                )
            else:
                st.markdown(
                    "Hello! I'm SupportAI. Analyze a complaint and I can explain the "
                    "issue, priority, escalation risk and recommended resolution. "
                    "You can also ask me how this system works."
                )

    # History
    for message in messages:
        avatar = "🤖" if message["role"] == "assistant" else "🧑‍💼"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Answer the last user message (streamed, with a loading indicator)
    if messages and messages[-1]["role"] == "user":
        user_text = messages[-1]["content"]

        with st.chat_message("assistant", avatar="🤖"):
            instant = None
            if contains_credentials(user_text):
                response = CREDENTIAL_WARNING
                st.markdown(response)
            elif (instant := instant_answer(user_text)) is not None:
                response = instant
                st.markdown(response)
            else:
                try:
                    llm_messages = [
                        {"role": "system", "content": build_system_prompt(user_text)}
                    ] + messages[-4:]

                    with st.spinner("SupportAI is thinking..."):
                        response = st.write_stream(stream_llm(llm_messages))
                    response = str(response)
                except Exception as exc:
                    response = llm_error_message(exc)
                    st.markdown(response)

        final = response if instant else sanitize_response(response)
        messages.append({"role": "assistant", "content": final})
        rerun_chat()

    # Quick questions
    questions = get_quick_questions(include_system_questions)
    cols = st.columns(quick_cols)

    for i, (label, question, needs_analysis) in enumerate(questions):
        with cols[i % quick_cols]:
            st.button(
                label,
                key=f"{key}_quick_{i}",
                use_container_width=True,
                disabled=needs_analysis and analysis is None,
                on_click=queue_user_message,
                args=(question,),
                help="Analyze a complaint first" if needs_analysis and analysis is None else None,
            )

    if messages:
        st.button(
            "🗑️ Clear Chat",
            key=f"{key}_clear",
            use_container_width=True,
            on_click=clear_chat,
        )

    # Free-text input
    chat_input = st.chat_input("Ask about the complaint, priority or escalation...", key=f"{key}_input")

    if chat_input:
        queue_user_message(chat_input.strip())
        rerun_chat()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🤖</div>
            <div class="sidebar-title">SupportAI</div>
            <div class="sidebar-subtitle">Customer Support Intelligence</div>
        </div>
        """
    )

    for section_title, items in NAV_SECTIONS:

        render_html(f'<div class="sidebar-section">{section_title}</div>')

        for icon, name in items:
            st.button(
                f"{icon}  {name}",
                key=f"nav_{name}",
                use_container_width=True,
                type="primary" if st.session_state.page == name else "secondary",
                on_click=go_to,
                args=(name,),
            )

    render_html(
        f"""
        <div class="sidebar-section">DASHBOARD DATA</div>
        """
    )

    st.button(
        "🗑️  Reset dashboard stats",
        key="reset_stats",
        use_container_width=True,
        on_click=reset_stats,
        help="Clears the saved analysis log and the KPI cards (handy before a demo).",
    )

    render_html(
        """
        <div class="sidebar-footer">
            <div>● SYSTEM ONLINE</div>
            <span>Local AI • ML • NLP</span>
        </div>
        """
    )


# ============================================================
# TOP HEADER
# ============================================================

_online = ollama_online()
if _online:
    warm_up_llm()
_assistant_badge = (
    '<div class="system-status status-online"><span class="status-dot"></span>ASSISTANT ONLINE</div>'
    if _online
    else '<div class="system-status status-offline"><span class="status-dot"></span>ASSISTANT OFFLINE</div>'
)

render_html(
    f"""
    <div class="top-header">
        <div>
            <div class="brand-title">Customer Support <span>Intelligence</span></div>
            <div class="brand-subtitle">
                AI-powered complaint analysis and intelligent resolution support
            </div>
        </div>
        <div class="header-badges">
            <div class="system-status status-online"><span class="status-dot"></span>ML ENGINE ONLINE</div>
            {_assistant_badge}
        </div>
    </div>
    """
)


# ============================================================
# PAGE: COMPLAINT ANALYSIS
# ============================================================

page = st.session_state.page

if page == "Complaint Analysis":

    page_header(
        "Analyze Customer Complaint",
        "Enter a customer complaint and let the AI identify the issue, assess "
        "priority, calculate escalation risk, and recommend a resolution.",
    )

    kpi_slot = st.container()  # filled after the analysis below, so it is always current

    spacer(18)

    left_col, right_col = st.columns([1.55, 0.95], gap="large")

    with left_col:

        render_html('<div class="section-label">Customer Complaint</div>')

        complaint = st.text_area(
            "Customer Complaint",
            key="complaint_input",
            height=130,
            placeholder=(
                "Example: My phone battery is draining very quickly "
                "and the device shuts down even when it has 30% battery..."
            ),
            label_visibility="collapsed",
        )

        # quick-fill sample complaints
        render_html('<div class="section-label" style="margin-top:6px;">Try a sample</div>')
        sample_cols = st.columns(len(SAMPLE_COMPLAINTS))
        for col, (label, text) in zip(sample_cols, SAMPLE_COMPLAINTS.items()):
            with col:
                st.button(
                    label,
                    key=f"sample_{label}",
                    use_container_width=True,
                    on_click=load_sample,
                    args=(text,),
                )

        analyze_clicked = st.button(
            "🔍  Analyze Complaint", type="primary", use_container_width=True
        )

        if analyze_clicked:
            if not complaint.strip():
                st.warning("Please enter a customer complaint before analysis.")
            else:
                with st.spinner("Analyzing complaint..."):
                    result = analyze_complaint(complaint.strip())
                st.session_state.analysis_result = result
                st.session_state.history.insert(0, result)
                st.session_state.history = st.session_state.history[:25]
                log_analysis(result)
                st.toast(f"Analysis complete: {result['issue']} · {result['priority']} priority", icon="✅")

        with kpi_slot:
            render_kpi_row()

        result = st.session_state.analysis_result

        if result:

            issue = result["issue"]
            priority = result["priority"]
            resolution = result["resolution"]
            escalation = result["escalation"]

            priority_class = PRIORITY_CLASS.get(priority, "low-value")
            risk_class = RISK_CLASS.get(escalation["risk"], "risk-low")

            spacer()

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                render_html(result_card("Detected Issue", issue, "Semantic issue classification", "issue-value"))
            with c2:
                render_html(result_card("Priority", priority, "Severity assessment", priority_class))
            with c3:
                render_html(
                    result_card(
                        "Risk Level",
                        escalation["risk"],
                        f"Risk score: {escalation['score']}",
                        risk_class,
                    )
                )
            with c4:
                render_html(
                    result_card(
                        "Next Action",
                        escalation["action"],
                        f"Route to: {result['resolution_team']}",
                        risk_class,
                    )
                )

            spacer()

            level = result["resolution_level"]
            level_class = {"Critical": "lvl-critical", "Elevated": "lvl-elevated"}.get(level, "lvl-standard")

            render_html(
                f"""
                <div class="resolution-card">
                    <div class="res-head">
                        <div class="resolution-title" style="margin-bottom:0;">💡 Recommended Resolution</div>
                        <span class="level-badge {level_class}">{esc(level)} response</span>
                    </div>

                    <div class="resolution-text">{esc(resolution)}</div>

                    <div class="res-footer">
                        <div class="res-meta">
                            <div class="res-meta-label">Recommended Action</div>
                            <div class="res-meta-value {risk_class}">→ {esc(escalation["action"])}</div>
                        </div>
                        <div class="res-meta">
                            <div class="res-meta-label">Route To</div>
                            <div class="res-meta-value">{esc(result["resolution_team"])}</div>
                        </div>
                    </div>

                    <div class="res-chips">
                        <span class="res-chip">Issue: <b class="issue-value">{esc(issue)}</b></span>
                        <span class="res-chip">Priority: <b class="{priority_class}">{esc(priority)}</b></span>
                        <span class="res-chip">Risk: <b class="{risk_class}">{esc(escalation["risk"])}</b></span>
                    </div>
                </div>
                """
            )

            spacer()

            m_col, e_col = st.columns(2, gap="medium")

            with m_col:
                bars = ""
                for name, sim in result["top_matches"]:
                    pct = max(0, min(100, round(sim * 100)))
                    bars += f"""
                        <div class="match-row">
                            <div class="match-head"><span>{esc(name)}</span><span>{pct}%</span></div>
                            <div class="match-bar"><div class="match-fill" style="width:{pct}%"></div></div>
                        </div>
                    """
                render_html(
                    f"""
                    <div class="glass-card">
                        <div class="resolution-title">Top Issue Matches</div>
                        {bars}
                    </div>
                    """
                )

            with e_col:
                render_html(escalation_card_html(escalation))

            render_html(
                pipeline_html(
                    ["Complaint", "Semantic Issue", "Priority", "Escalation Risk", "Resolution"]
                )
            )

            spacer(12)

            pdf_col, txt_col = st.columns([2, 1], gap="small")
            with pdf_col:
                st.download_button(
                    "📄  Download PDF Report",
                    data=result["pdf"],
                    file_name=f"supportai_report_{result['timestamp']:%Y%m%d_%H%M%S}.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
            with txt_col:
                st.download_button(
                    "TXT",
                    data=build_report(result),
                    file_name=f"supportai_report_{result['timestamp']:%Y%m%d_%H%M%S}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        else:

            render_html(
                """
                <div class="glass-card" style="margin-top:20px;text-align:center;">
                    <div style="font-size:30px;margin-bottom:10px;">🧠</div>
                    <div style="font-size:15px;font-weight:800;color:#ffffff;">Ready to Analyze</div>
                    <div style="font-size:12px;color:#6f82a4;margin-top:6px;">
                        Enter a customer complaint above, or pick a sample, to start the AI analysis pipeline.
                    </div>
                </div>
                """
            )

        # Recent analyses
        if st.session_state.history:

            spacer()

            rows = ""
            for item in st.session_state.history[:6]:
                rows += f"""
                    <div class="history-row">
                        <span>{item['timestamp']:%H:%M}</span>
                        <span class="history-text">{esc(item['text'])}</span>
                        <span>{esc(item['issue'])}</span>
                        <span class="{PRIORITY_CLASS.get(item['priority'], '')}">{esc(item['priority'])}</span>
                        <span class="{RISK_CLASS.get(item['escalation']['risk'], '')}">{esc(item['escalation']['risk'])}</span>
                    </div>
                """

            render_html(
                f"""
                <div class="glass-card">
                    <div class="card-title">Recent Analyses</div>
                    <div class="history-row head">
                        <span>Time</span><span>Complaint</span><span>Issue</span><span>Priority</span><span>Risk</span>
                    </div>
                    {rows}
                </div>
                """
            )

        insights = insights_html(read_log())
        if insights:
            spacer()
            ins_left, ins_right = st.columns(2, gap="medium")
            with ins_left:
                render_html(insights[0])
            with ins_right:
                render_html(insights[1])

    with right_col:

        with st.container(border=True):

            render_html(
                """
                <div class="chat-header">
                    <div class="chat-avatar">🤖</div>
                    <div>
                        <div class="chat-name">AI Support Assistant</div>
                        <div class="chat-status">● Online • Local AI</div>
                    </div>
                </div>
                """
            )

            render_chat(key="chat_analysis", quick_cols=2)


# ============================================================
# PAGE: AI SUPPORT ASSISTANT
# ============================================================

elif page == "AI Support Assistant":

    page_header("AI Support Assistant", "Conversational support powered by a local Llama 3.2 model.")

    chat_col, info_col = st.columns([1.65, 0.75], gap="large")

    with chat_col:

        with st.container(border=True):

            render_html(
                """
                <div class="chat-header">
                    <div class="chat-avatar">🤖</div>
                    <div>
                        <div class="chat-name">SupportAI Assistant</div>
                        <div class="chat-status">● Online • Llama 3.2 3B • Local</div>
                    </div>
                </div>
                """
            )

            render_chat(key="chat_assistant", include_system_questions=True, quick_cols=3)

    with info_col:

        render_html(
            info_card(
                "Assistant",
                "The chatbot runs locally using Ollama and Llama 3.2 3B. "
                "No customer data leaves your machine.",
                value="Local AI",
            )
        )

        spacer()

        render_html(
            info_card(
                "Capabilities",
                "• Complaint discussion<br><br>• Troubleshooting guidance<br><br>"
                "• Priority explanation<br><br>• Escalation guidance<br><br>"
                "• Resolution assistance",
            )
        )


# ============================================================
# PAGE: ML INTELLIGENCE
# ============================================================

elif page == "ML Intelligence":

    page_header(
        "ML Intelligence",
        "How the SupportAI system works: from raw support tickets to a recommended action.",
    )

    try:
        embedding_dim = semantic_model.get_sentence_embedding_dimension() or 384
    except Exception:
        embedding_dim = 384

    overview = [
        ("8,469", "Support Tickets", "Training dataset", "issue-value"),
        ("17", "Features", "Columns per ticket", ""),
        (str(len(issue_prototypes)), "Issue Categories", "+ 4 priority levels", "high-value"),
        (str(embedding_dim), "Embedding Size", "Dimensions per complaint", ""),
        ("MiniLM", "Semantic NLP", "all-MiniLM-L6-v2", "low-value"),
    ]

    for column, (value, label, small, cls) in zip(st.columns(5, gap="medium"), overview):
        with column:
            render_html(result_card(label, value, small, cls))

    spacer(26)

    # ---------- pipeline ----------

    render_html('<div class="section-label">AI Processing Pipeline</div>')
    render_html(stage_grid_html())

    spacer(22)

    render_html('<div class="section-label">Stage details</div>')

    tabs = st.tabs([f"{n}. {stage['title']}" for n, stage in enumerate(PIPELINE_STAGES, 1)])

    for tab, stage in zip(tabs, PIPELINE_STAGES):
        with tab:
            render_html(
                f"""
                <div class="glass-card">
                    <div class="card-title">{stage["icon"]} {esc(stage["title"])}</div>
                    <div class="card-description" style="font-size:13px;">{stage["body"]}</div>
                    {pipeline_html(stage["flow"])}
                    {stage_extra_html(stage["title"])}
                </div>
                """
            )

    spacer(22)

    # ---------- model information + baseline ----------

    info_col, base_col = st.columns([1, 1.15], gap="large")

    with info_col:
        render_html(
            kv_card(
                "Model Information",
                [
                    ("Embedding model", "all-MiniLM-L6-v2"),
                    ("Embedding size", f"{embedding_dim} dimensions"),
                    ("Issue detection", "Prototype embeddings + cosine similarity"),
                    ("Priority engine", "Semantic similarity + critical override"),
                    ("Escalation engine", "Rule-based, fully explainable"),
                    ("Resolution engine", "Issue × severity playbook"),
                    ("AI assistant", "Llama 3.2 3B via Ollama (local)"),
                    ("Training data", "8,469 tickets · 17 features"),
                ],
            )
        )

    with base_col:
        render_html('<div class="section-label">Baseline vs final approach</div>')

        b1, b2 = st.columns(2, gap="medium")

        with b1:
            render_html(
                """
                <div class="glass-card">
                    <div class="card-title">Baseline · TF-IDF</div>
                    <div class="card-value">98.35%</div>
                    <div class="card-description">
                        Issue-classification accuracy on the ticket dataset. Strong on
                        ticket-style wording, but in testing it did not generalize well
                        to naturally written complaints.
                    </div>
                </div>
                """
            )

        with b2:
            render_html(
                """
                <div class="glass-card" style="border-color:rgba(100,140,255,0.35);">
                    <div class="card-title">Final · Semantic Prototypes</div>
                    <div class="card-value issue-value">Meaning-based</div>
                    <div class="card-description">
                        Compares what a complaint means rather than which words it uses,
                        so differently worded complaints still map to the right issue.
                    </div>
                </div>
                """
            )

        spacer(10)

        render_html(
            """
            <div class="chat-welcome">
                <b>Why this matters:</b> a high benchmark score does not guarantee
                real-world performance. Because the TF-IDF model struggled with
                naturally phrased complaints, the final system uses sentence
                embeddings. The live validation below checks it end to end.
            </div>
            """
        )

    spacer(22)

    # ---------- live validation ----------

    render_html(
        info_card(
            "Live Pipeline Validation",
            "Runs 10 typical complaints through the full pipeline (issue, priority, "
            "escalation, resolution) so you can verify the system end to end.",
        )
    )

    spacer(10)

    if st.button("🧪 Run live validation", type="primary", key="run_res_tests"):

        rows_html = ""
        passed = 0
        details = []

        with st.spinner("Running 10 test complaints..."):
            for text, expected in RESOLUTION_TESTS:
                r = analyze_complaint(text)
                ok = r["issue"] in expected
                passed += ok

                rows_html += f"""
                    <div class="test-row">
                        <span>{esc(text)}</span>
                        <span>{esc(" / ".join(sorted(expected)))}</span>
                        <span class="issue-value">{esc(r["issue"])}</span>
                        <span class="{PRIORITY_CLASS.get(r["priority"], "")}">{esc(r["priority"])}</span>
                        <span class="{RISK_CLASS.get(r["escalation"]["risk"], "")}">{esc(r["escalation"]["risk"])}</span>
                        <span>{"✅" if ok else "⚠️"}</span>
                    </div>
                """
                details.append((text, r))

        render_html(
            f"""
            <div class="glass-card">
                <div class="card-title">Results: {passed} / {len(RESOLUTION_TESTS)} issues matched the expected category</div>
                <div class="test-row head">
                    <span>Complaint</span><span>Expected</span><span>Detected</span>
                    <span>Priority</span><span>Risk</span><span>Result</span>
                </div>
                {rows_html}
            </div>
            """
        )

        with st.expander("View the recommended resolution for each test"):
            for text, r in details:
                st.markdown(
                    f"**{text}**  \n"
                    f"*{r['issue']} • {r['priority']} priority • {r['resolution_level']} response • "
                    f"Route to {r['resolution_team']}*  \n"
                    f"{r['resolution']}"
                )
                st.divider()


# ============================================================
# PAGE: SEMANTIC ISSUE DETECTION
# ============================================================

elif page == "Semantic Issue Detection":

    page_header(
        "Semantic Issue Detection",
        "Natural-language complaints are mapped to issue categories using "
        "sentence embeddings and cosine similarity.",
    )

    render_html(
        info_card(
            "Detection Method",
            "The complaint is transformed into a 384-dimensional sentence embedding "
            "using all-MiniLM-L6-v2. Each issue category has a prototype embedding "
            "generated from the training tickets. The closest prototype is selected "
            "using cosine similarity.",
        )
    )

    spacer()

    tags = "".join(f'<span class="tag">{esc(name)}</span>' for name in sorted(issue_prototypes.keys()))
    render_html(
        f"""
        <div class="glass-card">
            <div class="card-title">Supported Issue Categories</div>
            {tags}
        </div>
        """
    )

    spacer()

    test_issue = st.text_area(
        "Test semantic issue detection",
        placeholder="Example: I cannot access my account anymore...",
        height=100,
    )

    if st.button("⚡ Detect Issue", type="primary", key="test_issue"):
        if test_issue.strip():
            with st.spinner("Detecting issue..."):
                scores = get_issue_scores(test_issue.strip())

            bars = ""
            for name, sim in scores[:5]:
                pct = max(0, min(100, round(sim * 100)))
                bars += f"""
                    <div class="match-row">
                        <div class="match-head"><span>{esc(name)}</span><span>{pct}%</span></div>
                        <div class="match-bar"><div class="match-fill" style="width:{pct}%"></div></div>
                    </div>
                """

            spacer(8)
            r1, r2 = st.columns([1, 1.4], gap="medium")
            with r1:
                render_html(result_card("Predicted Issue", scores[0][0], "Best prototype match", "issue-value"))
            with r2:
                render_html(
                    f"""
                    <div class="glass-card">
                        <div class="card-title">Closest Categories</div>
                        {bars}
                    </div>
                    """
                )
        else:
            st.warning("Please enter some text to analyze.")


# ============================================================
# PAGE: PRIORITY PREDICTION
# ============================================================

elif page == "Priority Prediction":

    page_header(
        "Priority Prediction",
        "Semantic similarity is used to estimate the urgency and operational "
        "impact of a customer complaint.",
    )

    priorities = [
        ("Critical", "System-wide failure, serious security issue or critical data impact.", "critical-value"),
        ("High", "Important functionality has stopped working or strongly affects usage.", "high-value"),
        ("Medium", "A recurring or noticeable issue that does not completely block usage.", "medium-value"),
        ("Low", "General information, recommendation or non-urgent assistance.", "low-value"),
    ]

    for column, (level, desc, cls) in zip(st.columns(4), priorities):
        with column:
            render_html(result_card("Priority Level", level, desc, cls))

    spacer()

    priority_test = st.text_area(
        "Test priority prediction",
        placeholder="Example: The entire system is down and I cannot access anything...",
        height=100,
    )

    if st.button("🎯 Predict Priority", type="primary", key="test_priority"):
        if priority_test.strip():
            with st.spinner("Predicting priority..."):
                predicted_priority = predict_priority(priority_test.strip())

            spacer(8)
            render_html(
                result_card(
                    "Predicted Priority",
                    predicted_priority,
                    "Based on semantic similarity to reference severity examples",
                    PRIORITY_CLASS.get(predicted_priority, "low-value"),
                )
            )
        else:
            st.warning("Please enter some text to analyze.")


# ============================================================
# PAGE: ESCALATION RISK ENGINE
# ============================================================

elif page == "Escalation Risk Engine":

    page_header(
        "Escalation Risk Engine",
        "A transparent, rule-based decision layer that decides how urgently a human "
        "should handle a complaint, and shows exactly why.",
    )

    # risk levels
    levels = [
        ("Low", "Normal support workflow", "low-value"),
        ("Medium", "Priority support queue", "medium-value"),
        ("High", "Priority human escalation", "high-value"),
        ("Critical", "Immediate human escalation", "critical-value"),
    ]

    for column, (level, action, cls) in zip(st.columns(4, gap="medium"), levels):
        with column:
            render_html(result_card(f"Score {RISK_RANGES[level]}", level, action, cls))

    spacer(22)

    test_col, rules_col = st.columns([1.45, 1], gap="large")

    with test_col:

        render_html('<div class="section-label">Test the engine</div>')

        st.text_area(
            "Test escalation engine",
            key="esc_test_input",
            height=120,
            placeholder=(
                "Example: My account has been hacked and I cannot "
                "access my important data. This is urgent."
            ),
            label_visibility="collapsed",
        )

        sample_cols = st.columns(len(ESCALATION_SAMPLES))
        for col, (label, text) in zip(sample_cols, ESCALATION_SAMPLES.items()):
            with col:
                st.button(
                    label,
                    key=f"esc_sample_{label}",
                    use_container_width=True,
                    on_click=load_esc_sample,
                    args=(text,),
                )

        if st.button(
            "🚨 Calculate Escalation Risk",
            type="primary",
            key="test_escalation",
            use_container_width=True,
        ):
            text = st.session_state.esc_test_input.strip()

            if not text:
                st.warning("Please enter a complaint to analyze.")
            else:
                with st.spinner("Running analysis..."):
                    test_issue = predict_issue_semantic(text)
                    test_priority = predict_priority(text)
                    st.session_state.esc_test_result = {
                        "issue": test_issue,
                        "priority": test_priority,
                        "team": get_resolution_team(test_issue),
                        "escalation": escalation_risk_engine(text, test_issue, test_priority),
                    }

        test_result = st.session_state.esc_test_result

        if test_result:

            spacer(8)

            render_html(
                f"""
                <div class="chip-row">
                    <span class="res-chip">Issue: <b class="issue-value">{esc(test_result["issue"])}</b></span>
                    <span class="res-chip">Priority: <b class="{PRIORITY_CLASS.get(test_result["priority"], "")}">{esc(test_result["priority"])}</b></span>
                    <span class="res-chip">Route to: <b>{esc(test_result["team"])}</b></span>
                </div>
                """
            )

            render_html(escalation_card_html(test_result["escalation"]))

        else:

            render_html(
                """
                <div class="glass-card" style="margin-top:18px;text-align:center;">
                    <div style="font-size:28px;margin-bottom:8px;">🚨</div>
                    <div style="font-size:14px;font-weight:800;color:#fff;">Try the engine</div>
                    <div style="font-size:12px;color:#6f82a4;margin-top:6px;">
                        Pick a sample or type a complaint, then calculate the risk.
                    </div>
                </div>
                """
            )

    with rules_col:
        render_html(signal_rules_html())


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="app-footer">
        SupportAI • AI-Powered Customer Support Intelligence
        • Semantic ML • Priority Prediction • Escalation Risk Engine • Local AI
    </div>
    """
)