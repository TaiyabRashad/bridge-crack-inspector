import streamlit as st
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import requests
import os

st.set_page_config(
    page_title="Bridge Crack Inspector",
    page_icon="🏗️",
    layout="wide"
)

@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("models/v3.pt"):
        with st.spinner("Loading model — please wait..."):
            url = "https://github.com/TaiyabRashad/bridge-crack-inspector/raw/main/v3.pt"
            r = requests.get(url, stream=True)
            with open("models/v3.pt", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return YOLO("models/v3.pt")

def draw_results(image_path, results, conf_threshold):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.imshow(img)
    boxes = results[0].boxes
    count = 0
    for box in boxes:
        c = float(box.conf[0])
        if c < conf_threshold:
            continue
        count += 1
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        color = "#e74c3c" if c >= 0.8 else "#e67e22" if c >= 0.6 else "#f39c12"
        risk = "HIGH" if c >= 0.8 else "MED" if c >= 0.6 else "LOW"
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2.5, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1+4, y1+18, f"CRACK  {risk}  {c:.0%}",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=color, alpha=0.9, linewidth=0))
    ax.axis("off")
    plt.tight_layout()
    return fig, count

# ── Header ──────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://raw.githubusercontent.com/TaiyabRashad/bridge-crack-inspector/main/4dbde425-0407-4c72-9858-a4207df9e853.jpg", width=100)
with col_title:
    st.title("Bridge Crack Detection System")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("**Taiyab Rashad** | Student ID: 100928272629929")
        st.markdown("University of Strathclyde | BEng Civil Engineering")
    with col_b:
        st.markdown("**Version 1.0** | [View Code on GitHub](https://github.com/TaiyabRashad/bridge-crack-inspector)")
        st.markdown("*YOLOv11s | DMRB CS 450*")

st.divider()

# ── Main Layout ─────────────────────────────────────
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
    st.caption("⚠️ For research and academic use only. All inspections must be verified by a qualified structural engineer per DMRB CS 450.")

with col2:
    if uploaded and run_btn:
        img_path = f"/tmp/{uploaded.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Analysing image..."):
            model = load_model()
            results = model.predict(
                source=img_path,
                imgsz=640,
                conf=conf_threshold,
                verbose=False
            )

        fig, count = draw_results(img_path, results, conf_threshold)

        if count == 0:
            st.success("✅  CLEAR — No defects detected. Structure appears sound.")
        else:
            max_conf = max(float(b.conf[0]) for b in results[0].boxes)
            if max_conf >= 0.8:
                st.error("🚨  CRITICAL — Immediate engineer inspection required (DMRB CS 450)")
            elif max_conf >= 0.6:
                st.warning("⚠️  WARNING — Engineer inspection required. Do not clear structure.")
            else:
                st.warning("🟡  MONITOR — Low confidence detections. Schedule follow-up.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Cracks Detected", count)
        m2.metric("Confidence Threshold", f"{conf_threshold:.0%}")
        m3.metric("Inspection Time", datetime.now().strftime("%H:%M"))

        st.pyplot(fig)
        st.caption(f"📍 {location}  |  {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  DMRB CS 450  |  YOLOv11s  |  v1.0")

    elif not uploaded:
        st.info("Upload an image on the left to begin inspection")
