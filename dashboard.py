"""
dashboard.py — Streamlit Interactive Web Dashboard for Semantic Segmentation
=============================================================================

This interactive dashboard allows users to:
1. Upload and run inference on custom images.
2. Select mock dataset images to test the model immediately.
3. Upload driving videos and view real-time segmented video playback.
4. Dynamically adjust mask transparency (Alpha).
5. Toggle visibility of individual perception classes (e.g., Pedestrian-only visualization).
6. Generate Grad-CAM heatmaps interactively to explain predictions.

Run:
    streamlit run dashboard.py

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

import os
import tempfile
import time
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Import project modules
from config import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES
from predict import SegmentationPredictor
from explainability import SegmentationGradCAM
from utils import colorize_mask, create_overlay

# Set Page Config
st.set_page_config(
    page_title="Autonomous Driving Perception Dashboard",
    page_icon="Car",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 20px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .class-box {
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        color: white;
        text-align: center;
        display: inline-block;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor(model_name: str, checkpoint_path: str, image_size: Tuple[int, int]) -> SegmentationPredictor:
    """Cache model loading so it doesn't reload on every interaction."""
    return SegmentationPredictor(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
    )


# Sidebar Configuration
st.sidebar.title(" Model Configuration")

# Model Architecture
model_name = st.sidebar.selectbox(
    "Select Model Architecture",
    ["unet", "resnet_unet", "deeplabv3plus", "segformer"],
    index=0,
)

# Locate Checkpoints
checkpoint_dir = "./checkpoints"
available_checkpoints = []
if os.path.exists(checkpoint_dir):
    available_checkpoints = [
        os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pth")
    ]

if not available_checkpoints:
    st.sidebar.warning("No checkpoint checkpoints/*.pth files found. Please train a model first.")
    checkpoint_path = st.sidebar.text_input("Manual Checkpoint Path", "checkpoints/best_model.pth")
else:
    checkpoint_path = st.sidebar.selectbox("Select Model Checkpoint", available_checkpoints, index=0)

# Image size for processing
img_size_str = st.sidebar.selectbox("Inference Processing Size", ["128x128", "256x256", "512x512"], index=0)
h_size, w_size = map(int, img_size_str.split("x"))

# Visual Adjustments
st.sidebar.title(" Visual Adjustments")
alpha = st.sidebar.slider("Overlay Transparency (Alpha)", 0.0, 1.0, 0.4, 0.05)

# Class Toggles
st.sidebar.subheader("Filter Perception Classes")
selected_classes = st.sidebar.multiselect(
    "Only visualize selected classes (shows all if empty):",
    CLASS_NAMES,
    default=CLASS_NAMES,
)

# Header Section
st.markdown("<div class='main-title'>Car Self-Driving Car Perception Module</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Interactive Real-Time Semantic Road Segmentation Dashboard</div>",
    unsafe_allow_html=True,
)

# Render Class Legends
st.subheader("Perception Class Legends")
legend_cols = st.columns(len(CLASS_NAMES))
for idx, name in enumerate(CLASS_NAMES):
    r, g, b = CLASS_COLORS[idx]
    color_hex = f"rgb({r},{g},{b})"
    legend_cols[idx].markdown(
        f"<div class='class-box' style='background-color: {color_hex};'>{name}</div>",
        unsafe_allow_html=True,
    )

# Main Body Tabs
tab1, tab2, tab3 = st.tabs([
    " Image Segmentation & Explainability",
    " Video Segmentation Playback",
    " Live Webcam Perception"
])

# Check model availability before proceeding
predictor = None
if os.path.exists(checkpoint_path):
    try:
        predictor = load_predictor(model_name, checkpoint_path, (h_size, w_size))
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
else:
    st.error(f"Checkpoint file '{checkpoint_path}' does not exist. Please place your trained best_model.pth here.")

# Filter map builder based on sidebar settings
def get_custom_colored_mask(mask: np.ndarray) -> np.ndarray:
    """Generate colored mask applying class visibility filters."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, name in enumerate(CLASS_NAMES):
        if not selected_classes or name in selected_classes:
            color_mask[mask == class_idx] = CLASS_COLORS[class_idx]
        else:
            # Set unselected classes to dark background
            color_mask[mask == class_idx] = (0, 0, 0)
    return color_mask


# TAB 1: IMAGE SEGMENTATION
with tab1:
    st.header(" Image Segmentation")
    col_input, col_action = st.columns([1, 1])

    # Sample option list
    sample_dir = "./dataset/images/test"
    sample_files = []
    if os.path.exists(sample_dir):
        sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

    input_mode = col_input.radio("Choose Input Image Source:", ["Use Mock Dataset Image", "Upload Custom Image"])

    image_to_process = None

    if input_mode == "Use Mock Dataset Image":
        if not sample_files:
            col_input.warning("No mock test images found. Generate mock dataset first.")
        else:
            selected_sample = col_input.selectbox("Select Mock Test Image", sample_files, index=0)
            sample_path = os.path.join(sample_dir, selected_sample)
            image_to_process = cv2.imread(sample_path)
            image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)
    else:
        uploaded_file = col_input.file_uploader("Upload a Road Scene Image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image_to_process = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB)

    if image_to_process is not None:
        col_input.image(image_to_process, caption="Selected Input Image", use_container_width=True)

        if predictor is not None:
            if col_action.button(" Run Segmenter & Explainability", use_container_width=True):
                with st.spinner("Processing image and running model backprop..."):
                    # Step 1: Predict
                    results = predictor.predict(image_to_process)
                    mask = results["mask"]
                    confidence = results["confidence"]

                    # Apply custom class toggles
                    custom_colored_mask = get_custom_colored_mask(mask)
                    custom_overlay = cv2.addWeighted(image_to_process, 1 - alpha, custom_colored_mask, alpha, 0)

                    # Show outputs in 2x2 grid
                    st.success("Perception analysis complete!")
                    o_col1, o_col2 = st.columns(2)
                    
                    o_col1.image(custom_colored_mask, caption="Predicted Masks Map", use_container_width=True)
                    o_col2.image(custom_overlay, caption=f"Semantic Overlay (Alpha={alpha})", use_container_width=True)

                    o_col1_2, o_col2_2 = st.columns(2)
                    
                    # Color map for confidence
                    conf_uint8 = (confidence * 255).astype(np.uint8)
                    conf_colored = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)
                    conf_colored = cv2.cvtColor(conf_colored, cv2.COLOR_BGR2RGB)
                    o_col1_2.image(conf_colored, caption="Inference Pixel Confidence Heatmap (Jet Map)", use_container_width=True)

                    # Explainability: Grad-CAM
                    st.divider()
                    st.subheader(" Grad-CAM Explainability (Class Attention Heatmaps)")
                    st.info(
                        "Grad-CAM computes gradients w.r.t. specific classes to show which exact pixels "
                        "the model 'looked at' to verify safety-critical features (like pedestrians)."
                    )

                    gradcam = SegmentationGradCAM(predictor.model, device=predictor.device)
                    tensor_input = predictor.preprocess(image_to_process)[0]

                    # Generate attention grid for classes present
                    gc_cols = st.columns(4)
                    col_index = 0
                    
                    unique_classes = np.unique(mask)
                    for c_idx in unique_classes:
                        class_name = CLASS_NAMES[c_idx]
                        if not selected_classes or class_name in selected_classes:
                            heatmap = gradcam.generate(tensor_input, int(c_idx))
                            h, w = image_to_process.shape[:2]
                            heatmap_resized = cv2.resize(heatmap, (w, h))

                            # Create heatmap overlay
                            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
                            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                            gc_overlay = cv2.addWeighted(image_to_process, 0.6, heatmap_colored, 0.4, 0)

                            current_col = gc_cols[col_index % 4]
                            current_col.image(gc_overlay, caption=f"Attention Map: {class_name}", use_container_width=True)
                            col_index += 1


# TAB 2: VIDEO SEGMENTATION
with tab2:
    st.header(" Video Playback Segmentation")
    st.write(
        "Upload a road-scene driving video. The dashboard will process it frame-by-frame, "
        "apply the semantic segmentation maps live, and display the processed video."
    )

    video_file = st.file_uploader("Upload Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        tfile.close()

        # OpenCV Video Capture
        cap = cv2.VideoCapture(tfile.name)
        
        # Video Stats
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = orig_fps if orig_fps > 0 else 30.0
        st.write(f"Loaded Video Stats: Total Frames = **{total_frames}**, Original FPS = **{fps:.1f}**")

        # Playback container
        st_frame = st.image([])
        progress_bar = st.progress(0.0)
        fps_placeholder = st.empty()

        if predictor is not None:
            if st.button(" Start Video Segmentation", use_container_width=True):
                frame_idx = 0
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Predict frame
                    # Speed boost: downsample frame processing size, run inference, resize back
                    results = predictor.predict(frame_rgb)
                    mask = results["mask"]

                    # Custom class toggles and overlay
                    colored_mask = get_custom_colored_mask(mask)
                    overlay = cv2.addWeighted(frame_rgb, 1 - alpha, colored_mask, alpha, 0)

                    # Display frame in Streamlit
                    st_frame.image(overlay, use_container_width=True)

                    # Progress & Stats
                    frame_idx += 1
                    progress = frame_idx / total_frames
                    progress_bar.progress(progress)
                    
                    elapsed = time.time() - start_time
                    current_fps = frame_idx / max(elapsed, 1e-4)
                    fps_placeholder.metric("Processing Playback Speed", f"{current_fps:.1f} FPS")

                cap.release()
                os.unlink(tfile.name)
                st.success("Video processing complete!")


# TAB 3: LIVE WEBCAM PERCEPTION
with tab3:
    st.header(" Live Webcam Perception")
    st.write(
        "Capture snapshots from your browser webcam or run a live webcam feed stream using your local camera."
    )
    
    webcam_mode = st.radio("Choose Webcam Input Mode:", ["Live Camera Stream", "Webcam Snapshot"])
    
    if webcam_mode == "Live Camera Stream":
        st.subheader("Live Camera Stream")
        st.write("Click 'Start Stream' to capture and segment your webcam feed in real-time. Uncheck 'Active' or click 'Stop Stream' to end.")
        
        camera_id = st.number_input("Camera Device ID (typically 0 for built-in webcam)", min_value=0, max_value=10, value=0, step=1)
        
        col_start, col_stop = st.columns(2)
        start_pressed = col_start.button(" Start Stream", use_container_width=True)
        stop_pressed = col_stop.button(" Stop Stream", use_container_width=True)
        
        if "webcam_streaming" not in st.session_state:
            st.session_state.webcam_streaming = False
            
        if start_pressed:
            st.session_state.webcam_streaming = True
        if stop_pressed:
            st.session_state.webcam_streaming = False
            
        st_webcam_frame = st.image([])
        fps_webcam_placeholder = st.empty()
        
        if st.session_state.webcam_streaming and predictor is not None:
            cap_webcam = cv2.VideoCapture(int(camera_id))
            if not cap_webcam.isOpened():
                st.error(f"Cannot open webcam device ID {camera_id}. Please verify that a camera is connected.")
                st.session_state.webcam_streaming = False
            else:
                frame_count = 0
                start_time = time.time()
                try:
                    while st.session_state.webcam_streaming:
                        ret, frame = cap_webcam.read()
                        if not ret:
                            st.warning("Failed to receive frame from camera. Stopping.")
                            break
                            
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = predictor.predict(frame_rgb)
                        mask = results["mask"]
                        
                        colored_mask = get_custom_colored_mask(mask)
                        overlay = cv2.addWeighted(frame_rgb, 1 - alpha, colored_mask, alpha, 0)
                        
                        st_webcam_frame.image(overlay, use_container_width=True)
                        
                        frame_count += 1
                        elapsed = time.time() - start_time
                        fps = frame_count / max(elapsed, 1e-4)
                        fps_webcam_placeholder.metric("Webcam Inference Speed", f"{fps:.1f} FPS")
                        
                        # Short sleep to prevent CPU starvation and allow Streamlit state checks
                        time.sleep(0.01)
                finally:
                    cap_webcam.release()
                    st_webcam_frame.empty()
                    
    elif webcam_mode == "Webcam Snapshot":
        st.subheader("Webcam Snapshot")
        st.write("Snap a photo using your web browser's camera input to run prediction and attention explainability.")
        
        camera_photo = st.camera_input("Capture Road Scene Snapshot")
        
        if camera_photo is not None:
            img = Image.open(camera_photo)
            img_np = np.array(img)
            
            st.success("Snapshot captured!")
            
            if predictor is not None:
                with st.spinner("Analyzing snapshot..."):
                    results = predictor.predict(img_np)
                    mask = results["mask"]
                    confidence = results["confidence"]
                    
                    custom_colored_mask = get_custom_colored_mask(mask)
                    custom_overlay = cv2.addWeighted(img_np, 1 - alpha, custom_colored_mask, alpha, 0)
                    
                    o_col1, o_col2 = st.columns(2)
                    o_col1.image(custom_colored_mask, caption="Predicted Masks Map", use_container_width=True)
                    o_col2.image(custom_overlay, caption=f"Semantic Overlay (Alpha={alpha})", use_container_width=True)
                    
                    o_col1_2, o_col2_2 = st.columns(2)
                    conf_uint8 = (confidence * 255).astype(np.uint8)
                    conf_colored = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)
                    conf_colored = cv2.cvtColor(conf_colored, cv2.COLOR_BGR2RGB)
                    o_col1_2.image(conf_colored, caption="Inference Pixel Confidence Heatmap (Jet Map)", use_container_width=True)
                    
                    st.divider()
                    st.subheader(" Grad-CAM Explainability (Class Attention Heatmaps)")
                    
                    gradcam = SegmentationGradCAM(predictor.model, device=predictor.device)
                    tensor_input = predictor.preprocess(img_np)[0]
                    
                    gc_cols = st.columns(4)
                    col_index = 0
                    
                    unique_classes = np.unique(mask)
                    for c_idx in unique_classes:
                        class_name = CLASS_NAMES[c_idx]
                        if not selected_classes or class_name in selected_classes:
                            heatmap = gradcam.generate(tensor_input, int(c_idx))
                            h, w = img_np.shape[:2]
                            heatmap_resized = cv2.resize(heatmap, (w, h))
                            
                            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
                            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                            gc_overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
                            
                            current_col = gc_cols[col_index % 4]
                            current_col.image(gc_overlay, caption=f"Attention Map: {class_name}", use_container_width=True)
                            col_index += 1

