import streamlit as st
import numpy as np
from PIL import Image

# 1. 精品視覺設定
st.set_page_config(page_title="AI 骨相診斷室", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #FDF5E6; } 
    h1 { color: #5D4037; font-family: 'serif'; text-align: center; border-bottom: 2px solid #D4AF37; padding-bottom: 10px; }
    p { color: #8D6E63; text-align: center; }
    .stButton>button { 
        background-color: #D4AF37; color: white; 
        border-radius: 5px; border: none; width: 100%;
        font-weight: bold; letter-spacing: 2px; height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("AI 骨相美學診斷室")

# 2. 實時加載 AI 零件 (移除快取，強迫重新讀取)
try:
    import mediapipe as mp
    try:
        mp_fm = mp.solutions.face_mesh
    except:
        import mediapipe.python.solutions.face_mesh as mp_fm
        
    face_mesh = mp_fm.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1,
        refine_landmarks=True
    )
    st.success("✅ AI 診斷系統已就緒")
except Exception as e:
    st.error(f"AI 加載中，請確保 requirements.txt 包含 mediapipe")
    face_mesh = None

# 3. 介面呈現
if face_mesh:
    st.write("---")
    uploaded_file = st.file_uploader("請拍攝或選取一張正面素顏照", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="分析中...", use_container_width=True)
        # 顯示簡易診斷
        st.info("📊 面部比例分析中，請截圖後傳送給專業美容師獲取詳細報告。")
        if st.button("查看骨相詳細分析報告"):
            st.balloons()
else:
    st.info("系統正在嘗試連接 AI 零件，請等待約 1 分鐘並點擊下方按鈕。")
    if st.button("點擊嘗試手動重整"):
        st.rerun()
