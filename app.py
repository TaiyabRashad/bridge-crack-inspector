import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from huggingface_hub import hf_hub_download
import os
from PIL import Image

# ── Page config ─────────────────────────────────────
st.set_page_config(
    page_title="Bridge Crack Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS — dark engineering theme ─────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e13;
    color: #c9d1d9;
}
.stApp { background-color: #0a0e13; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 1400px; }

.top-banner {
    background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
    border-bottom: 1px solid #21262d;
    padding: 1rem 2rem;
    margin: -1rem -2rem 2rem -2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.banner-title { font-size: 1.1rem; font-weight: 600; color: #e6edf3; letter-spacing: 0.02em; }
.banner-subtitle { font-size: 0.75rem; color: #6e7681; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.banner-badge {
    margin-left: auto;
    background: #1f2937; border: 1px solid #374151;
    border-radius: 4px; padding: 4px 10px;
    font-size: 0.7rem; color: #9ca3af; font-family: 'JetBrains Mono', monospace;
}
.status-dot {
    width: 8px; height: 8px; background: #238636;
    border-radius: 50%; display: inline-block;
    margin-right: 6px; box-shadow: 0 0 6px #238636;
}
.section-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6e7681;
    margin-bottom: 0.6rem; font-family: 'JetBrains Mono', monospace;
}
.panel {
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 6px; padding: 1.25rem; margin-bottom: 1rem;
}
.metric-row { display: flex; gap: 0.75rem; margin: 1rem 0; }
.metric-card {
    flex: 1; background: #0d1117; border: 1px solid #21262d;
    border-radius: 6px; padding: 0.9rem 1rem; text-align: center;
}
.metric-value {
    font-size: 1.8rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1; margin-bottom: 4px;
}
.metric-label { font-size: 0.68rem; color: #6e7681; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-confirmed { color: #f85149; }
.metric-uncertain { color: #d29922; }
.metric-time { color: #58a6ff; }

.alert { border-radius: 6px; padding: 0.85rem 1rem; margin: 1rem 0; font-size: 0.875rem; font-weight: 500; border-left: 3px solid; }
.alert-critical { background: #1a0a0a; border-color: #f85149; color: #ffa198; }
.alert-warning   { background: #1a1200;  border-color: #d29922; color: #e3b341; }
.alert-monitor   { background: #0d1a0d;  border-color: #238636; color: #3fb950; }
.alert-uncertain { background: #0d1117;  border-color: #58a6ff; color: #79c0ff; }
.alert-clear     { background: #0d1a0d;  border-color: #238636; color: #3fb950; }

.stButton > button {
    background: #238636 !important; color: #ffffff !important;
    border: 1px solid #2ea043 !important; border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 0.875rem !important; padding: 0.6rem 1.25rem !important;
    width: 100%;
}
.stButton > button:hover { background: #2ea043 !important; }

.stTextInput input {
    background: #0d1117 !important; border: 1px solid #30363d !important;
    border-radius: 6px !important; color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
}
.stTextInput label, .stSlider label, .stFileUploader label {
    color: #8b949e !important; font-size: 0.75rem !important;
    font-weight: 500 !important; text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
hr { border-color: #21262d !important; margin: 1.25rem 0 !important; }

.conf-tag {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; padding: 2px 8px; border-radius: 3px; font-weight: 600;
}
.conf-high { background: #f851491a; color: #f85149; border: 1px solid #f8514933; }
.conf-med  { background: #d299221a; color: #d29922; border: 1px solid #d2992233; }
.conf-low  { background: #2386361a; color: #3fb950; border: 1px solid #23863633; }

.footer-text {
    font-size: 0.68rem; color: #484f58;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 1.5rem; padding-top: 1rem;
    border-top: 1px solid #21262d; line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("models/v2.pt"):
        hf_hub_download(repo_id="Tai-Rashad/concrete-crack-inspector", filename="v2.pt", local_dir="models")
    if not os.path.exists("models/v3.pt"):
        hf_hub_download(repo_id="Tai-Rashad/concrete-crack-inspector", filename="v3.pt", local_dir="models")
    return YOLO("models/v2.pt"), YOLO("models/v3.pt")


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def ensemble_detect(model_v2, model_v3, image_path, conf=0.35):
    r2 = model_v2.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
    r3 = model_v3.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
    b2 = r2[0].boxes.xyxy.cpu().numpy() if len(r2[0].boxes) > 0 else []
    b3 = r3[0].boxes.xyxy.cpu().numpy() if len(r3[0].boxes) > 0 else []
    c2 = r2[0].boxes.conf.cpu().numpy() if len(r2[0].boxes) > 0 else []
    c3 = r3[0].boxes.conf.cpu().numpy() if len(r3[0].boxes) > 0 else []
    confirmed, uncertain, matched = [], [], set()
    for i, box2 in enumerate(b2):
        hit = False
        for j, box3 in enumerate(b3):
            if j in matched: continue
            if calculate_iou(box2, box3) > 0.3:
                confirmed.append({"box": box2, "confidence": (c2[i]+c3[j])/2})
                matched.add(j); hit = True; break
        if not hit:
            uncertain.append({"box": box2, "confidence": c2[i]})
    for j, box3 in enumerate(b3):
        if j not in matched:
            uncertain.append({"box": box3, "confidence": c3[j]})
    return confirmed, uncertain


def draw_results(image_path, confirmed, uncertain):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(img)
    for det in confirmed:
        box, c = det["box"], det["confidence"]
        color = "#f85149" if c >= 0.8 else "#d29922" if c >= 0.6 else "#3fb950"
        risk  = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
        ax.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor=color, facecolor="none"))
        ax.text(box[0]+4, box[1]+16, f"CRACK · {risk} · {c:.0%}",
                color="white", fontsize=8, fontweight="bold", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color, alpha=0.92, linewidth=0))
    for det in uncertain:
        box, c = det["box"], det["confidence"]
        ax.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=1.2, edgecolor="#484f58", facecolor="none", linestyle="--"))
        ax.text(box[0]+4, box[1]+16, f"UNCERTAIN · {c:.0%}",
                color="#8b949e", fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#161b22", alpha=0.9, linewidth=0))
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── TOP BANNER ───────────────────────────────────────
st.markdown("""
<div class="top-banner">
    <img src="https://raw.githubusercontent.com/TaiyabRashad/bridge-crack-inspector/main/4dbde425-0407-4c72-9858-a4207df9e853.jpg"
         style="height:38px; border-radius:4px; border:1px solid #30363d;">
    <div>
        <div class="banner-title">Bridge Crack Inspector</div>
        <div class="banner-subtitle">YOLOv11 Dual-Model Ensemble &nbsp;·&nbsp; DMRB CS 450 &nbsp;·&nbsp; Rashad Co.</div>
    </div>
    <div class="banner-badge">
        <span class="status-dot"></span>V2 + V3 ONLINE
    </div>
</div>
""", unsafe_allow_html=True)


# ── LAYOUT ───────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown('<div class="section-label">Inspection Input</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"],
                                 help="Bridge or concrete surface photo")

    location = st.text_input("Structure ID / Location", placeholder="e.g. M8-Bridge-01 / Glasgow")

    conf_threshold = st.slider("Detection threshold", min_value=0.10, max_value=0.90,
                                value=0.35, step=0.05,
                                help="Lower = more sensitive. Raise to reduce false positives.")

    run_btn = st.button("⬡  Run Inspection", use_container_width=True)

    st.markdown("---")

    st.markdown("""
    <div class="section-label">System Status</div>
    <div class="panel" style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#8b949e; line-height:2.2;">
        <span style="color:#3fb950">■</span>&nbsp; Model V2 &nbsp;&nbsp;YOLOv11n · 96.25% P<br>
        <span style="color:#3fb950">■</span>&nbsp; Model V3 &nbsp;&nbsp;YOLOv11s · 97.87% P<br>
        <span style="color:#58a6ff">■</span>&nbsp; Ensemble &nbsp;IoU cross-verify · 0.3 thr.<br>
        <span style="color:#6e7681">■</span>&nbsp; Standard &nbsp;DMRB CS 450 / FHWA 0.3mm
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-text">
        Taiyab Rashad · BEng Civil Engineering<br>
        University of Strathclyde · 2026<br>
        For academic and research use only.<br>
        Results must be verified by a qualified structural engineer.
    </div>
    """, unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="section-label">Analysis Output</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        img_path = f"/tmp/{uploaded.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Running dual-model ensemble analysis..."):
            model_v2, model_v3 = load_models()
            confirmed, uncertain = ensemble_detect(model_v2, model_v3, img_path, conf=conf_threshold)

        # Status alert
        if len(confirmed) == 0 and len(uncertain) == 0:
            st.markdown('<div class="alert alert-clear">✓ &nbsp;CLEAR — No structural defects detected.</div>', unsafe_allow_html=True)
        elif len(confirmed) > 0:
            max_conf = max(d["confidence"] for d in confirmed)
            if max_conf >= 0.8:
                st.markdown('<div class="alert alert-critical">⚠ &nbsp;CRITICAL — Immediate structural engineer inspection required (DMRB CS 450 §6.3)</div>', unsafe_allow_html=True)
            elif max_conf >= 0.6:
                st.markdown('<div class="alert alert-warning">⚡ &nbsp;WARNING — Engineer inspection required. Do not clear structure.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert alert-monitor">◎ &nbsp;MONITOR — Low-confidence detections. Schedule follow-up inspection.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-uncertain">◈ &nbsp;UNCERTAIN — Single-model detections only. Human review required.</div>', unsafe_allow_html=True)

        # Metrics
        max_conf_val = max((d["confidence"] for d in confirmed), default=0)
        loc_display = location if location else "—"
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value metric-confirmed">{len(confirmed)}</div>
                <div class="metric-label">Confirmed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-uncertain">{len(uncertain)}</div>
                <div class="metric-label">Uncertain</div>
            </div>
            <div class="metric-card">
                <div class="metric-value metric-time" style="font-size:1.2rem;padding-top:0.3rem">{max_conf_val:.0%}</div>
                <div class="metric-label">Peak Conf.</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#8b949e;font-size:0.85rem;padding-top:0.45rem">{datetime.now().strftime("%H:%M")}</div>
                <div class="metric-label">Scan Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Result image
        fig = draw_results(img_path, confirmed, uncertain)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Detection log
        if confirmed or uncertain:
            st.markdown('<div class="section-label" style="margin-top:1rem">Detection Log</div>', unsafe_allow_html=True)
            log_html = '<div class="panel">'
            for i, det in enumerate(confirmed, 1):
                c = det["confidence"]
                rc = "conf-high" if c >= 0.8 else "conf-med" if c >= 0.6 else "conf-low"
                rl = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
                log_html += f'<span style="font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#8b949e">#{i:02d} CONFIRMED &nbsp;</span><span class="conf-tag {rc}">{rl} · {c:.1%}</span><br>'
            for i, det in enumerate(uncertain, 1):
                c = det["confidence"]
                log_html += f'<span style="font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#484f58">#{i:02d} UNCERTAIN &nbsp;</span><span class="conf-tag" style="background:#1f2937;color:#6e7681;border:1px solid #30363d">{c:.1%}</span><br>'
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)

        # Report footer
        st.markdown(f"""
        <div class="footer-text">
            Report generated: {datetime.now().strftime('%d %b %Y · %H:%M')} &nbsp;·&nbsp;
            Structure: {loc_display} &nbsp;·&nbsp;
            Threshold: {conf_threshold:.0%} &nbsp;·&nbsp;
            Ensemble: V2 + V3
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div class="panel" style="text-align:center; padding: 4rem 2rem; border-style:dashed; border-color:#21262d;">
            <div style="font-size:2.5rem; margin-bottom:1rem; opacity:0.15;">⬡</div>
            <div style="color:#484f58; font-family:'JetBrains Mono',monospace; font-size:0.78rem; line-height:2.2;">
                Upload an inspection image to begin analysis<br>
                Supports JPG · JPEG · PNG
            </div>
        </div>
        """, unsafe_allow_html=True)
