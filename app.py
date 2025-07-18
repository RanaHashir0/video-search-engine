import streamlit as st
import cv2, tempfile, base64, subprocess
import torch, open_clip, faiss
import numpy as np
from PIL import Image

# 👉 1. Load model once
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    return model, tokenizer, preprocess

model, tokenizer, preprocess = load_model()

# 👉 2. Extract frames + timestamps once per upload
@st.cache_data(show_spinner=False)
def extract_and_embed(video_bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video_bytes); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 3)  # every 3 seconds
    frames, timestamps, embeds = [], [], []

    cnt, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if cnt % interval == 0:
            frames.append(frame)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            saved += 1
        cnt += 1
    cap.release()

    # Batch encode
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    tensors = torch.cat([preprocess(img).unsqueeze(0) for img in imgs], dim=0)
    with torch.no_grad():
        emb = model.encode_image(tensors).cpu().numpy()
    faiss.normalize_L2(emb)
    return tmp.name, timestamps, emb

# 👉 3. Build index once per video uploads
@st.cache_data(show_spinner=False)
def build_index(embeds):
    dim = embeds.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embeds)
    return idx

# 👉 4. UI 👇
st.title("🎬 Fast Video Scene Search")

uploaded = st.file_uploader("Upload MP4 video", type="mp4")
if uploaded:
    video_bytes = uploaded.read()
    st.success("Video uploaded!")

    vid_path, timestamps, embeds = extract_and_embed(video_bytes)
    index = build_index(embeds)

    query = st.text_input("Describe the scene:")
    if st.button("Search"):
        tokens = tokenizer([query])
        with torch.no_grad():
            txt = model.encode_text(tokens).cpu().numpy()
        faiss.normalize_L2(txt)
        _, I = index.search(txt, 1)
        t = timestamps[I[0][0]]
        st.success(f"📍 Found scene at {int(t//60)}:{int(t%60):02d}")

        video_encoded = base64.b64encode(video_bytes).decode()
        st.markdown(
            f'<video controls autoplay width="720">'
            f'<source src="data:video/mp4;base64,{video_encoded}#t={int(t)}">'
            f'</video>',
            unsafe_allow_html=True
        )
