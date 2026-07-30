import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import gdown
import os
from PIL import Image

st.set_page_config(
    page_title="Bridge Crack Inspector",
    page_icon="",
    layout="wide"
)

# ── Download models from Google Drive ──────────────
@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists("models/v2.pt"):
        with st.spinner("Loading Model V2..."):
            gdown.download(
                "https://drive.google.com/uc?id=1UwS3AcyrdMWVPBHoRfnPnvb15IU5XZD_",
                "models/v2.pt", quiet=False
            )
    
    if not os.path.exists("models/v3.pt"):
        with st.spinner("Loading Model V3..."):
            gdown.download(
                "https://drive.google.com/uc?id=14zMD7HCZuRnMEXHRQ6WVXqqva0QkNhDu",
                "models/v3.pt", quiet=False
            )
    
    model_v2 = YOLO("models/v2.pt")
    model_v3 = YOLO("models/v3.pt")
    return model_v2, model_v3

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def ensemble_detect(model_v2, model_v3, image_path, conf=0.35):
    results_v2 = model_v2.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
    results_v3 = model_v3.predict(source=image_path, imgsz=640, conf=conf, verbose=False)
    boxes_v2 = results_v2[0].boxes.xyxy.cpu().numpy() if len(results_v2[0].boxes) > 0 else []
    boxes_v3 = results_v3[0].boxes.xyxy.cpu().numpy() if len(results_v3[0].boxes) > 0 else []
    conf_v2  = results_v2[0].boxes.conf.cpu().numpy() if len(results_v2[0].boxes) > 0 else []
    conf_v3  = results_v3[0].boxes.conf.cpu().numpy() if len(results_v3[0].boxes) > 0 else []
    confirmed = []
    uncertain = []
    matched_v3 = set()
    for i, box2 in enumerate(boxes_v2):
        matched = False
        for j, box3 in enumerate(boxes_v3):
            if j in matched_v3:
                continue
            if calculate_iou(box2, box3) > 0.3:
                avg_conf = (conf_v2[i] + conf_v3[j]) / 2
                confirmed.append({"box": box2, "confidence": avg_conf})
                matched_v3.add(j)
                matched = True
                break
        if not matched:
            uncertain.append({"box": box2, "confidence": conf_v2[i]})
    for j, box3 in enumerate(boxes_v3):
        if j not in matched_v3:
            uncertain.append({"box": box3, "confidence": conf_v3[j]})
    return confirmed, uncertain

def draw_results(image_path, confirmed, uncertain):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.imshow(img)
    for det in confirmed:
        box = det["box"]
        c   = det["confidence"]
        color = "#e74c3c" if c >= 0.8 else "#e67e22" if c >= 0.6 else "#f39c12"
        risk  = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=2.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(box[0]+4, box[1]+18, f"CRACK  {risk}  {c:.0%}",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=color, alpha=0.9, linewidth=0))
    for det in uncertain:
        box = det["box"]
        c   = det["confidence"]
        rect = patches.Rectangle(
            (box[0], box[1]), box[2]-box[0], box[3]-box[1],
            linewidth=1.5, edgecolor="#95a5a6",
            facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(box[0]+4, box[1]+18, f"UNCERTAIN  {c:.0%}",
                color="white", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#7f8c8d", alpha=0.88, linewidth=0))
    ax.axis("off")
    plt.tight_layout()
    return fig

# ── UI ──────────────────────────────────────────────
st.set_page_config(
    page_title=" Concrete Crack Inspector",
    page_icon="",
    layout="wide"
)

# Header
# ── UI ──────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])

with col_logo:
    st.image("47373aad-944f-44c8-82d8-290b7e0c6bac.jpg", width=100)

with col_title:
    st.title("Bridge Crack Detection System")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("**Taiyab Rashad** | Director | Business: 100928272629929")
        st.markdown("Copyright © [2026] [Taiyab Rashad]. All rights reserved.All content, design, and underlying source code on this website are protected under international copyright treaties, including the Berne Convention. Any unauthorized reproduction, distribution, or modification of this intellectual property is strictly prohibited and subject to legal action worldwide.")
    with col_b:
        st.markdown("**Version 1.0** | [View Code on GitHub](https://github.com/TaiyabRashad/bridge-crack-inspector)")
        st.markdown("*Dual Model YOLOv11 Ensemble | DMRB CS 450*")

st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader(
        "Upload a bridge or concrete inspection photo",
        type=["jpg", "jpeg", "png"]
    )
    location = st.text_input("Location / Structure ID", "Bridge Deck — Span 1")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
    run_btn = st.button("🔍  Run Inspection", use_container_width=True)
    
    st.divider()
    st.caption("For research and academic use only. All inspections must be verified by a qualified structural engineer per DMRB CS 450.")

with col2:
    if uploaded and run_btn:
        img_path = f"/tmp/{uploaded.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Running dual model analysis..."):
            model_v2, model_v3 = load_models()
            confirmed, uncertain = ensemble_detect(
                model_v2, model_v3, img_path, conf=conf_threshold
            )

        if len(confirmed) == 0 and len(uncertain) == 0:
            st.success("✅  CLEAR — No defects detected. Structure appears sound.")
        elif len(confirmed) > 0:
            max_conf = max(d["confidence"] for d in confirmed)
            if max_conf >= 0.8:
                st.error("🚨  CRITICAL — Immediate engineer inspection required (DMRB CS 450)")
            elif max_conf >= 0.6:
                st.warning("⚠️  WARNING — Engineer inspection required. Do not clear structure.")
            else:
                st.warning("🟡  MONITOR — Low confidence detections. Schedule follow-up.")
        else:
            st.info("❓  UNCERTAIN — Single model detections only. Human review required.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Confirmed Cracks", len(confirmed))
        m2.metric("Uncertain Detections", len(uncertain))
        m3.metric("Inspection Time", datetime.now().strftime("%H:%M"))

        fig = draw_results(img_path, confirmed, uncertain)
        st.pyplot(fig)

        st.caption(f" {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  DMRB CS 450  |  YOLOv11n + YOLOv11s  |  v1.0")

    elif not uploaded:
        st.info("Upload an image on the left to begin inspection")
