import streamlit as st
import os
import cv2
import torch
import open_clip
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
import tempfile
import base64

# --- Load Model ---
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.to("cpu").eval()
    return model, tokenizer, preprocess

model, tokenizer, preprocess = load_model()

# --- Extract Frames ---
def extract_frames(video_path, interval_sec=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)
    count, saved = 0, 0
    frames, timestamps = [], []

    while True:
        success, frame = cap.read()
        if not success:
            break
        if count % interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(img)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            saved += 1
        count += 1
    cap.release()
    return frames, timestamps

# --- Encode Frames ---
def encode_frames(frames):
    embeddings = []
    for img in frames:
        try:
            img_tensor = preprocess(img).unsqueeze(0).to("cpu")
            with torch.no_grad():
                emb = model.encode_image(img_tensor).cpu().numpy()
                faiss.normalize_L2(emb)
                embeddings.append(emb[0])
        except:
            continue
    return np.stack(embeddings)

# --- Search ---
def search_text(query, embeddings, timestamps, top_k=1):
    with torch.no_grad():
        tokens = tokenizer([query]).to("cpu")
        text_feat = model.encode_text(tokens).cpu().numpy()
        faiss.normalize_L2(text_feat)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        D, I = index.search(text_feat, top_k)

    best_idx = I[0][0]
    return timestamps[best_idx]

# --- Embed Video in HTML ---
def get_video_html(video_bytes, start_time):
    encoded = base64.b64encode(video_bytes).decode()
    return f"""
    <video width="720" height="400" controls autoplay>
        <source src="data:video/mp4;base64,{encoded}#t={int(start_time)}" type="video/mp4">
    </video>
    """

# --- Streamlit UI ---
st.title("🎬 Video Scene Search Engine")
st.write("Upload a video and search for a scene using natural language.")

video_file = st.file_uploader("📤 Upload MP4 Video", type=["mp4"])

if video_file is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_file.read())
    temp_video_path = temp_video.name

    st.success("✅ Video uploaded. Extracting frames...")

    frames, timestamps = extract_frames(temp_video_path, interval_sec=2)

    st.info(f"✅ {len(frames)} frames extracted")

    embeddings = encode_frames(frames)

    query = st.text_input("🔍 Describe the scene (e.g. 'a person talking on phone')")

    if st.button("Search"):
        if query.strip() == "":
            st.warning("Please enter a description.")
        else:
            matched_time = search_text(query, embeddings, timestamps)

            mins = int(matched_time // 60)
            secs = int(matched_time % 60)

            st.success(f"🎯 Scene found at {mins:02d}:{secs:02d}")

            with open(temp_video_path, "rb") as f:
                video_bytes = f.read()
                html = get_video_html(video_bytes, matched_time)
                st.markdown(html, unsafe_allow_html=True)
