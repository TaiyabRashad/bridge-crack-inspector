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

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #1c1f24; color: #d4d8df; }
.stApp { background-color: #1c1f24; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 0 2rem 0 !important; max-width: 100% !important; }

.main-content { padding: 1.5rem 2rem; }

.gold-line { height: 2px; background: linear-gradient(90deg, #c5a03c, #8a6d1a, transparent); width: 100%; }
.topbar {
    background: #16191e; border-bottom: 1px solid #2a2d33;
    padding: 0 1.75rem; height: 54px; display: flex;
    align-items: center; gap: 1rem; position: relative; overflow: hidden;
}
.topbar::after {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(197,160,60,0.02) 3px,rgba(197,160,60,0.02) 4px);
    pointer-events: none;
}
.brand-name { font-size: 14px; font-weight: 600; color: #e8e4d9; letter-spacing: 0.02em; }
.brand-sub  { font-size: 10px; color: #5a5e66; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }
.topbar-right { margin-left: auto; display: flex; align-items: center; gap: 12px; }
.status-pill { display: flex; align-items: center; gap: 6px; background: #1a1c20; border: 1px solid #2a2d33; border-radius: 4px; padding: 4px 10px; font-size: 10px; font-family: 'JetBrains Mono', monospace; color: #c5a03c; }
.status-dot { width: 6px; height: 6px; background: #c5a03c; border-radius: 50%; display: inline-block; margin-right: 2px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.ver-badge { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: #3d4148; border: 1px solid #2a2d33; border-radius: 3px; padding: 3px 8px; }

.sec-label { font-size: 9px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #3d4148; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; display: block; }

.model-toggle {
    background: #13151a; border: 1px solid #2a2d33;
    border-radius: 5px; padding: 10px 12px; margin-bottom: 6px;
    display: flex; align-items: center; justify-content: space-between;
}
.model-toggle-on  { border-color: #c5a03c44; }
.model-toggle-off { opacity: 0.45; }
.model-name { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #d4d8df; font-weight: 500; }
.model-meta { font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #5a5e66; margin-top: 2px; }

.warn-box { background: #1a1208; border: 1px solid #3d1a10; border-radius: 5px; padding: 8px 12px; font-size: 10px; font-family: 'JetBrains Mono', monospace; color: #b05050; margin-bottom: 8px; }

.sys-panel { background: #13151a; border: 1px solid #2a2d33; border-radius: 5px; padding: 10px 12px; }
.sys-row { display: flex; align-items: center; gap: 8px; font-size: 10px; font-family: 'JetBrains Mono', monospace; color: #5a5e66; padding: 3px 0; }
.dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; display: inline-block; }
.dot-gold { background: #c5a03c; } .dot-blue { background: #6e8bb5; } .dot-dim { background: #3d4148; }

.alert { border-radius: 5px; padding: 10px 14px; margin: 0.75rem 0; font-size: 12px; font-weight: 500; border-left: 3px solid; display: flex; align-items: center; gap: 8px; }
.alert-critical { background: #1a1710; border-color: #c5a03c; color: #d4b060; }
.alert-warning   { background: #1a1208; border-color: #b05050; color: #c07070; }
.alert-monitor   { background: #111814; border-color: #5a7a5a; color: #7a9a7a; }
.alert-uncertain { background: #13151a; border-color: #3d4148; color: #5a5e66; }
.alert-clear     { background: #111814; border-color: #5a7a5a; color: #7a9a7a; }
.alert-error     { background: #1a0808; border-color: #b05050; color: #c07070; }

.metric-row { display: flex; gap: 8px; margin: 0.75rem 0; }
.metric-card { flex: 1; background: #16191e; border: 1px solid #2a2d33; border-radius: 6px; padding: 12px; text-align: center; }
.metric-val { font-size: 22px; font-weight: 600; font-family: 'JetBrains Mono', monospace; line-height: 1; margin-bottom: 4px; }
.metric-lbl { font-size: 9px; color: #3d4148; text-transform: uppercase; letter-spacing: 0.1em; }
.mv-gold { color: #c5a03c; } .mv-red { color: #b05050; } .mv-dim { color: #5a5e66; font-size: 13px; padding-top: 4px; }

.log-panel { background: #13151a; border: 1px solid #2a2d33; border-radius: 5px; padding: 8px 12px; }
.log-row { display: flex; align-items: center; gap: 8px; padding: 5px 0; border-bottom: 1px solid #1c1f24; font-size: 10px; font-family: 'JetBrains Mono', monospace; }
.log-row:last-child { border-bottom: none; }
.log-id { color: #3d4148; min-width: 24px; } .log-conf { color: #5a5e66; margin-left: auto; }
.tag { border-radius: 3px; padding: 1px 6px; font-size: 9px; font-weight: 600; border: 1px solid; }
.tag-high { background: #1a1710; color: #c5a03c; border-color: #3d3010; }
.tag-med  { background: #1a1208; color: #b05050; border-color: #3d1a10; }
.tag-unc  { background: #1c1f24; color: #3d4148; border-color: #2a2d33; font-weight: 400; }
.vote-badge { font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #5a5e66; background: #1c1f24; border: 1px solid #2a2d33; border-radius: 3px; padding: 1px 5px; }

.footer-strip { font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #2a2d33; border-top: 1px solid #2a2d33; padding: 7px 0; display: flex; gap: 16px; margin-top: 0.5rem; flex-wrap: wrap; }

.stButton > button {
    background: linear-gradient(135deg, #8a6d1a, #c5a03c) !important;
    color: #0d0f12 !important; border: none !important; border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 700 !important;
    font-size: 12px !important; padding: 0.65rem 1.25rem !important;
    letter-spacing: 0.04em !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.9 !important; }
.stFileUploader > div { background: #13151a !important; border: 1px dashed #2a2d33 !important; border-radius: 6px !important; }
.stTextInput input { background: #13151a !important; border: 1px solid #2a2d33 !important; border-radius: 5px !important; color: #d4d8df !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }
.stTextInput input:focus { border-color: #c5a03c !important; }
.stTextInput label, .stSlider label, .stFileUploader label { color: #3d4148 !important; font-size: 9px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; font-family: 'JetBrains Mono', monospace !important; }
.stCheckbox label { color: #d4d8df !important; font-size: 12px !important; font-family: 'JetBrains Mono', monospace !important; }
hr { border-color: #2a2d33 !important; margin: 1rem 0 !important; }
.stSpinner > div { border-top-color: #c5a03c !important; }
.empty-state { background: #13151a; border: 1px dashed #2a2d33; border-radius: 6px; text-align: center; padding: 4rem 2rem; }
.empty-icon { font-size: 2.5rem; opacity: 0.1; color: #c5a03c; margin-bottom: 1rem; }
.empty-text { color: #3d4148; font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 2; }

section[data-testid="stSidebar"] { padding: 0 !important; }
div[data-testid="stVerticalBlock"] { gap: 0.75rem; }
div[data-testid="column"]:first-child { padding: 1.5rem 1rem 1.5rem 2rem !important; }
div[data-testid="column"]:last-child  { padding: 1.5rem 2rem 1.5rem 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load all 3 models ────────────────────────────────
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    for fname in ["v2.pt", "v3.pt", "v4.pt"]:
        if not os.path.exists(f"models/{fname}"):
            with st.spinner(f"Loading {fname}..."):
                hf_hub_download(
                    repo_id="Tai-Rashad/concrete-crack-inspector",
                    filename=fname, local_dir="models"
                )
    return YOLO("models/v2.pt"), YOLO("models/v3.pt"), YOLO("models/v4.pt")


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def run_ensemble(models_dict, image_path, conf=0.35):
    """
    models_dict: {"V2": model_v2, "V3": model_v3} etc — only active models
    Confirmed = 2+ models agree (or 1 model if only 1 active)
    Uncertain = 1 model only (when 2+ active)
    """
    active_names = list(models_dict.keys())
    n_active = len(active_names)

    all_detections = []
    for name, model in models_dict.items():
        r = model.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
        if len(r[0].boxes) > 0:
            boxes = r[0].boxes.xyxy.cpu().numpy()
            confs = r[0].boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                all_detections.append({"box": box, "conf": confs[i], "model": name})

    # Cluster by IoU
    merged = []
    used = set()
    for i, det_a in enumerate(all_detections):
        if i in used:
            continue
        cluster = [det_a]
        used.add(i)
        for j, det_b in enumerate(all_detections):
            if j in used:
                continue
            if calculate_iou(det_a["box"], det_b["box"]) > 0.3:
                cluster.append(det_b)
                used.add(j)
        merged.append(cluster)

    confirmed = []
    uncertain = []

    for cluster in merged:
        models_agreeing = set(d["model"] for d in cluster)
        avg_conf = sum(d["conf"] for d in cluster) / len(cluster)
        best_box = cluster[0]["box"]

        # If only 1 model active — everything is confirmed
        # If 2+ active — need 2+ to agree for confirmed
        if n_active == 1 or len(models_agreeing) >= 2:
            confirmed.append({
                "box": best_box,
                "confidence": avg_conf,
                "votes": len(models_agreeing),
                "models": sorted(list(models_agreeing))
            })
        else:
            uncertain.append({
                "box": best_box,
                "confidence": avg_conf,
                "votes": 1,
                "models": sorted(list(models_agreeing))
            })

    confirmed.sort(key=lambda x: x["confidence"], reverse=True)
    uncertain.sort(key=lambda x: x["confidence"], reverse=True)
    return confirmed, uncertain


def draw_results(image_path, confirmed, uncertain):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#13151a")
    ax.set_facecolor("#13151a")
    ax.imshow(img)
    for det in confirmed:
        box, c = det["box"], det["confidence"]
        votes = det["votes"]
        color = "#c5a03c" if c >= 0.6 else "#8a6d1a"
        risk  = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
        lw = 2.5 if votes >= 3 else 2.0
        ax.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=lw, edgecolor=color, facecolor="none"))
        suffix = " · ✓✓✓" if votes >= 3 else ""
        ax.text(box[0]+4, box[1]+16, f"CRACK · {risk} · {c:.0%}{suffix}",
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

# Load models in background
model_v2, model_v3, model_v4 = load_models()

# Build active model label for topbar
def get_status_label(u2, u3, u4):
    active = [v for v, on in [("V2", u2), ("V3", u3), ("V4", u4)] if on]
    return " + ".join(active) + " ONLINE" if active else "NO MODELS ACTIVE"

# ── LAYOUT ──────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    # Model toggles
    st.markdown('<span class="sec-label">Model Selection</span>', unsafe_allow_html=True)

    use_v2 = st.checkbox("Model V2 — YOLOv11n · 96.25% P", value=True)
    use_v3 = st.checkbox("Model V3 — YOLOv11s · 97.87% P", value=True)
    use_v4 = st.checkbox("Model V4 — YOLOv11s · 78.10% P · 13,498 img", value=True)

    active_models = {}
    if use_v2: active_models["V2"] = model_v2
    if use_v3: active_models["V3"] = model_v3
    if use_v4: active_models["V4"] = model_v4

    n_active = len(active_models)

    # Mode description
    if n_active == 0:
        st.markdown('<div class="warn-box">⚠ No models selected. Enable at least one to run inspection.</div>', unsafe_allow_html=True)
    elif n_active == 1:
        st.markdown(f'<div class="sys-panel" style="margin-bottom:8px"><div class="sys-row"><span class="dot dot-gold"></span>Single model mode · {list(active_models.keys())[0]} only</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sys-panel" style="margin-bottom:8px"><div class="sys-row"><span class="dot dot-blue"></span>Ensemble mode · {n_active} models · 2/3 vote threshold</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<span class="sec-label">Inspection Input</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    location = st.text_input("Structure ID / Location", placeholder="e.g. M8-Bridge-04 / Glasgow")
    conf_threshold = st.slider("Detection threshold", 0.10, 0.90, 0.35, 0.05)
    run_btn = st.button("⬡  Run Inspection", disabled=(n_active == 0))

    st.markdown("---")
    st.markdown("""
    <div style="font-size:9px;font-family:'JetBrains Mono',monospace;color:#2a2d33;line-height:2">
        For research and academic use only.<br>
        Results must be verified by a qualified structural engineer.<br>
        DMRB CS 450 / FHWA 0.3mm threshold
    </div>
    """, unsafe_allow_html=True)

# Render topbar with live model status
status_label = get_status_label(use_v2, use_v3, use_v4)
mode_label = "Triple Ensemble" if n_active == 3 else "Dual Ensemble" if n_active == 2 else "Single Model" if n_active == 1 else "Offline"
st.markdown(f"""
<div class="topbar">
    <img src="https://raw.githubusercontent.com/TaiyabRashad/bridge-crack-inspector/main/4dbde425-0407-4c72-9858-a4207df9e853.jpg"
         style="height:34px;border-radius:4px;border:1px solid #2a2d33;flex-shrink:0"
         onerror="this.style.display='none'">
    <div>
        <div class="brand-name">Bridge Crack Inspector</div>
        <div class="brand-sub">Rashad Co. &nbsp;·&nbsp; YOLOv11 {mode_label} &nbsp;·&nbsp; DMRB CS 450</div>
    </div>
    <div class="topbar-right">
        <div class="status-pill"><span class="status-dot"></span>{status_label}</div>
        <div class="ver-badge">v4.1</div>
    </div>
</div>
""", unsafe_allow_html=True)

with col_right:
    if n_active == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-text">Enable at least one model on the left to begin</div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded and run_btn:
        img_path = f"/tmp/{uploaded.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        mode_str = f"{'+'.join(active_models.keys())} ensemble" if n_active > 1 else f"{list(active_models.keys())[0]} solo"
        with st.spinner(f"Running {mode_str} analysis..."):
            confirmed, uncertain = run_ensemble(active_models, img_path, conf=conf_threshold)

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
        triple_votes = sum(1 for d in confirmed if d["votes"] >= 3)
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
                models_str = "+".join(det["models"])
                log_html += f'<div class="log-row"><span class="log-id">#{i:02d}</span><span class="tag {tc}">{rl}</span><span style="color:#5a5e66">Confirmed · bbox [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]</span><span class="vote-badge">{models_str}</span><span class="log-conf">{c:.1%}</span></div>'
            for i, det in enumerate(uncertain, 1):
                c = det["confidence"]
                box = det["box"]
                models_str = det["models"][0] if det["models"] else "?"
                log_html += f'<div class="log-row"><span class="log-id">#{i:02d}</span><span class="tag tag-unc">UNCERTAIN</span><span style="color:#3d4148">Single model · bbox [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]</span><span class="vote-badge" style="color:#2a2d33">{models_str}</span><span class="log-conf" style="color:#3d4148">{c:.1%}</span></div>'
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)

        # Footer
        st.markdown(f"""
        <div class="footer-strip">
            <span>Report: {datetime.now().strftime('%d %b %Y · %H:%M')}</span>
            <span>Structure: {loc_display}</span>
            <span>Threshold: {conf_threshold:.0%}</span>
            <span>Active: {'+'.join(active_models.keys())}</span>
            <span>Mode: {mode_label}</span>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⬡</div>
            <div class="empty-text">Upload an inspection image to begin<br>JPG · JPEG · PNG</div>
        </div>
        """, unsafe_allow_html=True)
