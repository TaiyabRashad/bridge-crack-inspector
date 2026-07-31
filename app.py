import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from huggingface_hub import hf_hub_download
import os

st.set_page_config(
    page_title="Bridge Crack Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #1c1f24;
    color: #d4d8df;
}
.stApp { background-color: #1c1f24; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.gold-line {
    height: 2px;
    background: linear-gradient(90deg, #c5a03c, #8a6d1a, transparent);
    width: 100%;
}
.topbar {
    background: #16191e;
    border-bottom: 1px solid #2a2d33;
    padding: 0 1.75rem;
    height: 54px;
    display: flex;
    align-items: center;
    gap: 1rem;
    position: relative;
    overflow: hidden;
}
.topbar::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(197,160,60,0.02) 3px,rgba(197,160,60,0.02) 4px);
    pointer-events: none;
}
.brand-name { font-size: 14px; font-weight: 600; color: #e8e4d9; letter-spacing: 0.02em; }
.brand-sub  { font-size: 10px; color: #5a5e66; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.topbar-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
.status-pill {
    display: flex; align-items: center; gap: 6px;
    background: #1a1c20; border: 1px solid #2a2d33;
    border-radius: 4px; padding: 4px 10px;
    font-size: 10px; font-family: 'JetBrains Mono', monospace; color: #c5a03c;
}
.status-dot {
    width: 6px; height: 6px; background: #c5a03c;
    border-radius: 50%; display: inline-block; margin-right: 2px;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.ver-badge {
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: #3d4148; border: 1px solid #2a2d33;
    border-radius: 3px; padding: 3px 8px;
}
.sec-label {
    font-size: 9px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #3d4148;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; display: block;
}
.sys-panel {
    background: #13151a; border: 1px solid #2a2d33;
    border-radius: 5px; padding: 10px 12px;
}
.sys-row {
    display: flex; align-items: center; gap: 8px;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: #5a5e66; padding: 3px 0; line-height: 1.4;
}
.dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; display: inline-block; }
.dot-gold { background: #c5a03c; }
.dot-blue { background: #6e8bb5; }
.dot-dim  { background: #3d4148; }

.alert {
    border-radius: 5px; padding: 10px 14px;
    margin: 0.75rem 0; font-size: 12px; font-weight: 500;
    border-left: 3px solid; display: flex; align-items: center; gap: 8px;
}
.alert-critical { background: #1a1710; border-color: #c5a03c; color: #d4b060; }
.alert-warning   { background: #1a1208; border-color: #b05050; color: #c07070; }
.alert-monitor   { background: #111814; border-color: #5a7a5a; color: #7a9a7a; }
.alert-uncertain { background: #13151a; border-color: #3d4148; color: #5a5e66; }
.alert-clear     { background: #111814; border-color: #5a7a5a; color: #7a9a7a; }

.metric-row { display: flex; gap: 8px; margin: 0.75rem 0; }
.metric-card {
    flex: 1; background: #16191e; border: 1px solid #2a2d33;
    border-radius: 6px; padding: 12px; text-align: center;
}
.metric-val {
    font-size: 22px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1; margin-bottom: 4px;
}
.metric-lbl { font-size: 9px; color: #3d4148; text-transform: uppercase; letter-spacing: 0.1em; }
.mv-gold { color: #c5a03c; }
.mv-red  { color: #b05050; }
.mv-dim  { color: #5a5e66; font-size: 13px; padding-top: 4px; }

.log-panel { background: #13151a; border: 1px solid #2a2d33; border-radius: 5px; padding: 8px 12px; }
.log-row {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0; border-bottom: 1px solid #1c1f24;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
}
.log-row:last-child { border-bottom: none; }
.log-id { color: #3d4148; min-width: 24px; }
.log-conf { color: #5a5e66; margin-left: auto; }
.tag {
    border-radius: 3px; padding: 1px 6px;
    font-size: 9px; font-weight: 600; border: 1px solid;
}
.tag-high { background: #1a1710; color: #c5a03c; border-color: #3d3010; }
.tag-med  { background: #1a1208; color: #b05050; border-color: #3d1a10; }
.tag-unc  { background: #1c1f24; color: #3d4148; border-color: #2a2d33; font-weight: 400; }

.footer-strip {
    font-size: 9px; font-family: 'JetBrains Mono', monospace;
    color: #2a2d33; border-top: 1px solid #2a2d33;
    padding: 7px 0; display: flex; gap: 16px; margin-top: 0.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #8a6d1a, #c5a03c) !important;
    color: #0d0f12 !important; border: none !important;
    border-radius: 6px !important; font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important; font-size: 12px !important;
    padding: 0.65rem 1.25rem !important; letter-spacing: 0.04em !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

.stFileUploader > div {
    background: #13151a !important;
    border: 1px dashed #2a2d33 !important;
    border-radius: 6px !important;
}
.stTextInput input {
    background: #13151a !important; border: 1px solid #2a2d33 !important;
    border-radius: 5px !important; color: #d4d8df !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important;
}
.stTextInput input:focus { border-color: #c5a03c !important; box-shadow: 0 0 0 1px #c5a03c22 !important; }
.stTextInput label, .stSlider label, .stFileUploader label {
    color: #3d4148 !important; font-size: 9px !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; font-family: 'JetBrains Mono', monospace !important;
}
hr { border-color: #2a2d33 !important; margin: 1rem 0 !important; }
.stSpinner > div { border-top-color: #c5a03c !important; }

.empty-state {
    background: #13151a; border: 1px dashed #2a2d33;
    border-radius: 6px; text-align: center; padding: 4rem 2rem;
}
.empty-icon { font-size: 2.5rem; opacity: 0.1; color: #c5a03c; margin-bottom: 1rem; }
.empty-text { color: #3d4148; font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 2; }
</style>
""", unsafe_allow_html=True)


# ── Model loading — V3 + V4 ensemble ────────────────
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("models/v3.pt"):
        with st.spinner("Loading Model V3..."):
            hf_hub_download(repo_id="Tai-Rashad/concrete-crack-inspector", filename="v3.pt", local_dir="models")
    if not os.path.exists("models/v4.pt"):
        with st.spinner("Loading Model V4..."):
            hf_hub_download(repo_id="Tai-Rashad/concrete-crack-inspector", filename="v4.pt", local_dir="models")
    return YOLO("models/v3.pt"), YOLO("models/v4.pt")


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def ensemble_detect(model_v3, model_v4, image_path, conf=0.35):
    r3 = model_v3.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
    r4 = model_v4.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
    b3 = r3[0].boxes.xyxy.cpu().numpy() if len(r3[0].boxes) > 0 else []
    b4 = r4[0].boxes.xyxy.cpu().numpy() if len(r4[0].boxes) > 0 else []
    c3 = r3[0].boxes.conf.cpu().numpy() if len(r3[0].boxes) > 0 else []
    c4 = r4[0].boxes.conf.cpu().numpy() if len(r4[0].boxes) > 0 else []
    confirmed, uncertain, matched = [], [], set()
    for i, box3 in enumerate(b3):
        hit = False
        for j, box4 in enumerate(b4):
            if j in matched: continue
            if calculate_iou(box3, box4) > 0.3:
                confirmed.append({"box": box3, "confidence": (c3[i]+c4[j])/2})
                matched.add(j); hit = True; break
        if not hit:
            uncertain.append({"box": box3, "confidence": c3[i]})
    for j, box4 in enumerate(b4):
        if j not in matched:
            uncertain.append({"box": box4, "confidence": c4[j]})
    return confirmed, uncertain


def draw_results(image_path, confirmed, uncertain):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#13151a")
    ax.set_facecolor("#13151a")
    ax.imshow(img)
    for det in confirmed:
        box, c = det["box"], det["confidence"]
        color = "#c5a03c" if c >= 0.6 else "#8a6d1a"
        risk  = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
        ax.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2, edgecolor=color, facecolor="none"))
        ax.text(box[0]+4, box[1]+16, f"CRACK · {risk} · {c:.0%}",
                color="#0d0f12", fontsize=8, fontweight="bold", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color, alpha=0.95, linewidth=0))
    for det in uncertain:
        box, c = det["box"], det["confidence"]
        ax.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=1.2, edgecolor="#3d4148", facecolor="none", linestyle="--"))
        ax.text(box[0]+4, box[1]+16, f"UNCERTAIN · {c:.0%}",
                color="#5a5e66", fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1c1f24", alpha=0.9, linewidth=0))
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── TOP BAR ─────────────────────────────────────────
st.markdown('<div class="gold-line"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="topbar">
    <img src="https://raw.githubusercontent.com/TaiyabRashad/bridge-crack-inspector/main/4dbde425-0407-4c72-9858-a4207df9e853.jpg"
         style="height:34px;border-radius:4px;border:1px solid #2a2d33;flex-shrink:0"
         onerror="this.style.display='none'">
    <div>
        <div class="brand-name">Bridge Crack Inspector</div>
        <div class="brand-sub">Rashad Co. &nbsp;·&nbsp; YOLOv11 Ensemble &nbsp;·&nbsp; DMRB CS 450</div>
    </div>
    <div class="topbar-right">
        <div class="status-pill"><span class="status-dot"></span>V3 + V4 ONLINE</div>
        <div class="ver-badge">v4.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ──────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown('<span class="sec-label">Inspection Input</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    location = st.text_input("Structure ID / Location", placeholder="e.g. M8-Bridge-04 / Glasgow")
    conf_threshold = st.slider("Detection threshold", 0.10, 0.90, 0.35, 0.05)
    run_btn = st.button("⬡  Run Inspection")

    st.markdown("---")

    st.markdown("""
    <span class="sec-label">System Status</span>
    <div class="sys-panel">
        <div class="sys-row"><span class="dot dot-gold"></span>Model V3 &nbsp;&nbsp;YOLOv11s · 97.87% P</div>
        <div class="sys-row"><span class="dot dot-gold"></span>Model V4 &nbsp;&nbsp;YOLOv11s · 78.10% P · 13,498 img</div>
        <div class="sys-row"><span class="dot dot-blue"></span>Ensemble &nbsp;IoU cross-verify · 0.3 thr.</div>
        <div class="sys-row"><span class="dot dot-dim"></span>Standard &nbsp;DMRB CS 450 / FHWA 0.3mm</div>
    </div>
    <div style="margin-top:1rem;font-size:9px;font-family:'JetBrains Mono',monospace;color:#2a2d33;line-height:2">
        For research and academic use only.<br>
        Results must be verified by a qualified structural engineer.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    if uploaded and run_btn:
        img_path = f"/tmp/{uploaded.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Running V3 + V4 ensemble analysis..."):
            model_v3, model_v4 = load_models()
            confirmed, uncertain = ensemble_detect(model_v3, model_v4, img_path, conf=conf_threshold)

        # Alert
        if len(confirmed) == 0 and len(uncertain) == 0:
            st.markdown('<div class="alert alert-clear">✓ &nbsp;CLEAR — No structural defects detected. Structure appears sound.</div>', unsafe_allow_html=True)
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
            <div class="metric-card"><div class="metric-val mv-red">{len(confirmed)}</div><div class="metric-lbl">Confirmed</div></div>
            <div class="metric-card"><div class="metric-val mv-gold">{len(uncertain)}</div><div class="metric-lbl">Uncertain</div></div>
            <div class="metric-card"><div class="metric-val mv-gold">{max_conf_val:.0%}</div><div class="metric-lbl">Peak Conf.</div></div>
            <div class="metric-card"><div class="metric-val mv-dim">{datetime.now().strftime("%H:%M")}</div><div class="metric-lbl">Scan Time</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Result image
        fig = draw_results(img_path, confirmed, uncertain)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Detection log
        if confirmed or uncertain:
            st.markdown('<span class="sec-label" style="margin-top:0.75rem;display:block">Detection Log</span>', unsafe_allow_html=True)
            log_html = '<div class="log-panel">'
            for i, det in enumerate(confirmed, 1):
                c = det["confidence"]
                tc = "tag-high" if c >= 0.6 else "tag-med"
                rl = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
                box = det["box"]
                log_html += f'<div class="log-row"><span class="log-id">#{i:02d}</span><span class="tag {tc}">{rl}</span><span style="color:#5a5e66">Confirmed crack · bbox [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]</span><span class="log-conf">{c:.1%}</span></div>'
            for i, det in enumerate(uncertain, 1):
                c = det["confidence"]
                box = det["box"]
                log_html += f'<div class="log-row"><span class="log-id">#{i:02d}</span><span class="tag tag-unc">UNCERTAIN</span><span style="color:#3d4148">Single-model detection · bbox [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]</span><span class="log-conf" style="color:#3d4148">{c:.1%}</span></div>'
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)

        # Footer
        st.markdown(f"""
        <div class="footer-strip">
            <span>Report: {datetime.now().strftime('%d %b %Y · %H:%M')}</span>
            <span>Structure: {loc_display}</span>
            <span>Threshold: {conf_threshold:.0%}</span>
            <span>Ensemble: V3 + V4</span>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-text">Upload an inspection image to begin<br>JPG · JPEG · PNG</div>
        </div>
        """, unsafe_allow_html=True)
