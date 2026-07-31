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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f5f5f3;
    color: #1a1a1a;
}
.stApp { background-color: #f5f5f3; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.gold-bar { height: 3px; background: linear-gradient(90deg, #c5a03c 0%, #e8c96a 50%, #f5f5f3 100%); }

.topbar {
    background: #fff; border-bottom: 1px solid #e8e8e5;
    padding: 0 2rem; height: 56px;
    display: flex; align-items: center; gap: 1rem;
}
.brand-name { font-size: 14px; font-weight: 600; color: #1a1a1a; letter-spacing: -0.01em; }
.brand-sub  { font-size: 11px; color: #9a9a97; margin-top: 1px; }
.topbar-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
.status-pill {
    display: flex; align-items: center; gap: 6px;
    background: #faf8f0; border: 1px solid #e8d89a;
    border-radius: 20px; padding: 4px 12px;
    font-size: 11px; color: #8a6d1a; font-weight: 500;
}
.status-dot { width: 6px; height: 6px; background: #c5a03c; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.ver-badge { font-size: 11px; color: #9a9a97; background: #f0f0ee; border-radius: 4px; padding: 3px 8px; border: 1px solid #e8e8e5; }

.sec-label { font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #b0b0ad; margin-bottom: 10px; display: block; }

.mode-badge { background: #f0f8f0; border: 1px solid #c0dac0; border-radius: 6px; padding: 7px 12px; font-size: 11px; color: #4a7a4a; margin-top: 10px; margin-bottom: 4px; }
.mode-badge-warn { background: #fff8f0; border: 1px solid #f0d0a0; border-radius: 6px; padding: 7px 12px; font-size: 11px; color: #8a5a1a; margin-top: 10px; }

.alert { border-radius: 8px; padding: 12px 16px; margin-bottom: 1rem; font-size: 13px; font-weight: 500; display: flex; align-items: center; gap: 10px; border: 1px solid; }
.alert-critical { background: #fffbf0; border-color: #e8d89a; color: #8a6d1a; }
.alert-warning   { background: #fff5f5; border-color: #f0c0c0; color: #a04040; }
.alert-monitor   { background: #f5faf5; border-color: #b0d8b0; color: #3a6a3a; }
.alert-uncertain { background: #f8f8f6; border-color: #d8d8d4; color: #6a6a67; }
.alert-clear     { background: #f5faf5; border-color: #b0d8b0; color: #3a6a3a; }

.metrics { display: flex; gap: 10px; margin-bottom: 1rem; }
.metric-card { flex: 1; background: #fff; border: 1px solid #e8e8e5; border-radius: 8px; padding: 14px; text-align: center; }
.metric-val { font-size: 24px; font-weight: 600; line-height: 1; margin-bottom: 3px; }
.metric-lbl { font-size: 10px; color: #9a9a97; text-transform: uppercase; letter-spacing: 0.08em; }
.mv-gold { color: #c5a03c; }
.mv-red  { color: #c05050; }
.mv-dim  { color: #9a9a97; font-size: 15px; padding-top: 4px; }

.log-panel { background: #fff; border: 1px solid #e8e8e5; border-radius: 8px; overflow: hidden; }
.log-header { padding: 8px 14px; border-bottom: 1px solid #f0f0ee; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #b0b0ad; }
.log-row { display: flex; align-items: center; gap: 8px; padding: 8px 14px; border-bottom: 1px solid #f5f5f3; font-size: 11px; }
.log-row:last-child { border-bottom: none; }
.log-id { color: #b0b0ad; min-width: 24px; }
.log-conf { color: #9a9a97; margin-left: auto; }
.tag { border-radius: 4px; padding: 2px 7px; font-size: 10px; font-weight: 600; border: 1px solid; }
.tag-high { background: #faf8f0; color: #8a6d1a; border-color: #e8d89a; }
.tag-med  { background: #fff5f5; color: #a04040; border-color: #f0c0c0; }
.tag-unc  { background: #f5f5f3; color: #9a9a97; border-color: #e8e8e5; font-weight: 400; }
.vote-pill { font-size: 10px; color: #b0b0ad; background: #f5f5f3; border: 1px solid #e8e8e5; border-radius: 3px; padding: 1px 5px; }

.footer-strip { padding: 10px 0; border-top: 1px solid #e8e8e5; font-size: 10px; color: #b0b0ad; display: flex; gap: 16px; margin-top: 0.5rem; flex-wrap: wrap; }

.empty-state { background: #fff; border: 1.5px dashed #e0e0dc; border-radius: 10px; text-align: center; padding: 5rem 2rem; }
.empty-icon { font-size: 2.5rem; opacity: 0.08; margin-bottom: 1rem; }
.empty-text { color: #b0b0ad; font-size: 12px; line-height: 2; }

/* Streamlit overrides */
.stButton > button {
    background: #1a1a1a !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 13px !important; padding: 0.7rem 1.25rem !important;
    width: 100% !important; letter-spacing: -0.01em !important;
}
.stButton > button:hover { background: #2a2a2a !important; }
.stButton > button:disabled { background: #d0d0cc !important; color: #9a9a97 !important; }

.stFileUploader > div {
    background: #fafaf8 !important;
    border: 1.5px dashed #d8d8d4 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

.stTextInput input {
    background: #fafaf8 !important; border: 1px solid #e8e8e5 !important;
    border-radius: 6px !important; color: #1a1a1a !important;
    font-family: 'Inter', sans-serif !important; font-size: 12px !important;
    padding: 8px 10px !important;
}
.stTextInput input:focus { border-color: #c5a03c !important; box-shadow: 0 0 0 2px #c5a03c22 !important; }
.stTextInput input::placeholder { color: #b0b0ad !important; }

.stTextInput label, .stSlider label {
    color: #b0b0ad !important; font-size: 10px !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

.stCheckbox label p { color: #1a1a1a !important; font-size: 12px !important; font-weight: 500 !important; }
.stCheckbox { background: #fafaf8; border: 1px solid #e8e8e5; border-radius: 8px; padding: 8px 12px !important; margin-bottom: 6px !important; }

hr { border-color: #e8e8e5 !important; margin: 1.25rem 0 !important; }
.stSpinner > div { border-top-color: #c5a03c !important; }

div[data-testid="column"]:first-child {
    background: #fff;
    border-right: 1px solid #e8e8e5;
    padding: 1.5rem 1.5rem !important;
    min-height: calc(100vh - 59px);
}
div[data-testid="column"]:last-child {
    padding: 1.5rem 2rem !important;
    background: #f5f5f3;
}
</style>
""", unsafe_allow_html=True)


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
    n_active = len(models_dict)
    all_detections = []
    for name, model in models_dict.items():
        r = model.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
        if len(r[0].boxes) > 0:
            boxes = r[0].boxes.xyxy.cpu().numpy()
            confs = r[0].boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                all_detections.append({"box": box, "conf": confs[i], "model": name})

    merged = []
    used = set()
    for i, det_a in enumerate(all_detections):
        if i in used: continue
        cluster = [det_a]; used.add(i)
        for j, det_b in enumerate(all_detections):
            if j in used: continue
            if calculate_iou(det_a["box"], det_b["box"]) > 0.3:
                cluster.append(det_b); used.add(j)
        merged.append(cluster)

    confirmed, uncertain = [], []
    for cluster in merged:
        models_agreeing = set(d["model"] for d in cluster)
        avg_conf = sum(d["conf"] for d in cluster) / len(cluster)
        best_box = cluster[0]["box"]
        if n_active == 1 or len(models_agreeing) >= 2:
            confirmed.append({"box": best_box, "confidence": avg_conf, "votes": len(models_agreeing), "models": sorted(list(models_agreeing))})
        else:
            uncertain.append({"box": best_box, "confidence": avg_conf, "votes": 1, "models": sorted(list(models_agreeing))})

    confirmed.sort(key=lambda x: x["confidence"], reverse=True)
    uncertain.sort(key=lambda x: x["confidence"], reverse=True)
    return confirmed, uncertain


def draw_results(image_path, confirmed, uncertain):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#fafaf8")
    ax.set_facecolor("#fafaf8")
    ax.imshow(img)
    for det in confirmed:
        box, c = det["box"], det["confidence"]
        color = "#c5a03c" if c >= 0.6 else "#8a6d1a"
        risk  = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
        lw = 2.5 if det["votes"] >= 3 else 2.0
        ax.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=lw, edgecolor=color, facecolor=color+"22"))
        label = f"CRACK · {risk} · {c:.0%}"
        ax.text(box[0]+4, box[1]+16, label,
                color="#fff", fontsize=8, fontweight="bold", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color, alpha=1, linewidth=0))
    for det in uncertain:
        box, c = det["box"], det["confidence"]
        ax.add_patch(patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=1.2, edgecolor="#b0b0ad", facecolor="none", linestyle="--"))
        ax.text(box[0]+4, box[1]+16, f"UNCERTAIN · {c:.0%}",
                color="#6a6a67", fontsize=8, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f0ee", alpha=0.95, linewidth=0))
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── GOLD BAR + TOPBAR ───────────────────────────────
st.markdown('<div class="gold-bar"></div>', unsafe_allow_html=True)

model_v2, model_v3, model_v4 = load_models()

col_left, col_right = st.columns([1, 2], gap="small")

with col_left:
    st.markdown('<span class="sec-label">Model selection</span>', unsafe_allow_html=True)

    use_v2 = st.checkbox("Model V2 — YOLOv11n · 96.25% P", value=True)
    use_v3 = st.checkbox("Model V3 — YOLOv11s · 97.87% P", value=True)
    use_v4 = st.checkbox("Model V4 — YOLOv11s · 78.10% P · 13,498 img", value=True)

    active_models = {}
    if use_v2: active_models["V2"] = model_v2
    if use_v3: active_models["V3"] = model_v3
    if use_v4: active_models["V4"] = model_v4
    n_active = len(active_models)

    if n_active == 0:
        st.markdown('<div class="mode-badge-warn">⚠ No models selected</div>', unsafe_allow_html=True)
    elif n_active == 1:
        name = list(active_models.keys())[0]
        st.markdown(f'<div class="mode-badge">Single model mode · {name} only</div>', unsafe_allow_html=True)
    elif n_active == 2:
        names = " + ".join(active_models.keys())
        st.markdown(f'<div class="mode-badge">Dual ensemble · {names} · 2/2 vote</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-badge">Triple ensemble · V2 + V3 + V4 · 2/3 vote</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="sec-label">Inspection input</span>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], label_visibility="collapsed")
    location = st.text_input("Structure ID / Location", placeholder="e.g. M8-Bridge-04 / Glasgow")
    conf_threshold = st.slider("Detection threshold", 0.10, 0.90, 0.35, 0.05)
    run_btn = st.button("Run inspection", disabled=(n_active == 0))

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px;color:#b0b0ad;line-height:2">
        For research and academic use only.<br>
        All results must be verified by a qualified structural engineer.<br>
        DMRB CS 450 · FHWA 0.3mm threshold · Rashad Co.
    </div>
    """, unsafe_allow_html=True)

# ── TOPBAR rendered after columns so it overlays correctly ──
active_label = " + ".join(active_models.keys()) + " online" if active_models else "offline"
mode_label = "Triple Ensemble" if n_active == 3 else "Dual Ensemble" if n_active == 2 else "Single Model" if n_active == 1 else "Offline"
st.markdown(f"""
<div class="topbar">
    <img src="https://raw.githubusercontent.com/TaiyabRashad/bridge-crack-inspector/main/4dbde425-0407-4c72-9858-a4207df9e853.jpg"
         style="height:34px;border-radius:6px;border:1px solid #e8e8e5;flex-shrink:0"
         onerror="this.style.display='none'">
    <div>
        <div class="brand-name">Bridge Crack Inspector</div>
        <div class="brand-sub">Rashad Co. · YOLOv11 {mode_label} · DMRB CS 450</div>
    </div>
    <div class="topbar-right">
        <div class="status-pill"><span class="status-dot"></span>{active_label}</div>
        <div class="ver-badge">v4.1</div>
    </div>
</div>
""", unsafe_allow_html=True)

with col_right:
    if n_active == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">◎</div>
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
            st.markdown('<div class="alert alert-clear">✓ &nbsp;Clear — no structural defects detected. Structure appears sound.</div>', unsafe_allow_html=True)
        elif len(confirmed) > 0:
            max_conf = max(d["confidence"] for d in confirmed)
            if max_conf >= 0.8:
                st.markdown('<div class="alert alert-critical">⚠ &nbsp;Critical — immediate structural engineer inspection required (DMRB CS 450 §6.3)</div>', unsafe_allow_html=True)
            elif max_conf >= 0.6:
                st.markdown('<div class="alert alert-warning">⚡ &nbsp;Warning — engineer inspection required. Do not clear structure.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert alert-monitor">◎ &nbsp;Monitor — low-confidence detections. Schedule follow-up inspection.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-uncertain">◈ &nbsp;Uncertain — single-model detections only. Human review required.</div>', unsafe_allow_html=True)

        # Metrics
        max_conf_val = max((d["confidence"] for d in confirmed), default=0)
        triple_votes = sum(1 for d in confirmed if d["votes"] >= 3)
        loc_display = location if location else "—"

        st.markdown(f"""
        <div class="metrics">
            <div class="metric-card"><div class="metric-val mv-red">{len(confirmed)}</div><div class="metric-lbl">Confirmed</div></div>
            <div class="metric-card"><div class="metric-val mv-gold">{len(uncertain)}</div><div class="metric-lbl">Uncertain</div></div>
            <div class="metric-card"><div class="metric-val mv-gold">{max_conf_val:.0%}</div><div class="metric-lbl">Peak conf.</div></div>
            <div class="metric-card"><div class="metric-val mv-dim">{datetime.now().strftime("%H:%M")}</div><div class="metric-lbl">Scan time</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Result image
        fig = draw_results(img_path, confirmed, uncertain)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Detection log
        if confirmed or uncertain:
            log_html = '<div class="log-panel"><div class="log-header">Detection log</div>'
            for i, det in enumerate(confirmed, 1):
                c = det["confidence"]
                tc = "tag-high" if c >= 0.6 else "tag-med"
                rl = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
                box = det["box"]
                mv = "+".join(det["models"])
                log_html += f'<div class="log-row"><span class="log-id">#{i:02d}</span><span class="tag {tc}">{rl}</span><span style="color:#6a6a67">Confirmed crack · bbox [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]</span><span class="vote-pill">{mv}</span><span class="log-conf">{c:.1%}</span></div>'
            for i, det in enumerate(uncertain, 1):
                c = det["confidence"]
                box = det["box"]
                mv = det["models"][0] if det["models"] else "?"
                log_html += f'<div class="log-row"><span class="log-id">#{i:02d}</span><span class="tag tag-unc">UNCERTAIN</span><span style="color:#b0b0ad">Single model · bbox [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]</span><span class="vote-pill">{mv}</span><span class="log-conf" style="color:#b0b0ad">{c:.1%}</span></div>'
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)

        # Footer
        st.markdown(f"""
        <div class="footer-strip">
            <span>Report: {datetime.now().strftime('%d %b %Y · %H:%M')}</span>
            <span>Structure: {loc_display}</span>
            <span>Threshold: {conf_threshold:.0%}</span>
            <span>Active: {'+'.join(active_models.keys())} · {mode_label}</span>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">◎</div>
            <div class="empty-text">Upload an inspection image to begin analysis<br>JPG · JPEG · PNG supported</div>
        </div>
        """, unsafe_allow_html=True)
