import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# 1. 高級感精品視覺設定
st.set_page_config(page_title="AI 骨相診斷室", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #FDF5E6; } /* 米杏色背景 */
    h1, h2, h3 { color: #4A3728; font-family: 'serif'; } /* 深啡色文字 */
    .stButton>button { 
        background-color: #D4AF37; color: white; 
        border-radius: 25px; border: none; width: 100%;
        font-weight: bold; height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 初始化 AI 模型 (修正路徑相容性)
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except:
    try:
        import mediapipe.solutions.face_mesh as mp_face_mesh
    except:
        st.error("AI 模組加載失敗，請檢查 requirements.txt")

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


st.title("💊 AI 骨相美學診斷室")
st.write("透過 AI 偵測面部核心數據，為您量身打造原生感妝容方案。")

# 3. 上傳功能
uploaded_file = st.file_uploader("請拍攝或上傳一張正面素顏照", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    # 進行 AI 分析
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 視覺化偵測點
        annotated_image = img_array.copy()
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 2, (212, 175, 55), -1) # 金色偵測點
        
        col1, col2 = st.columns(2)
        with col1: st.image(image, caption="原始照片")
        with col2: st.image(annotated_image, caption="AI 骨相偵測圖")

        # 4. 骨相邏輯計算
        # 三庭：額頭(10), 眉心(168), 鼻尖(1), 下巴(152)
        u_third = landmarks[168].y - landmarks[10].y
        m_third = landmarks[1].y - landmarks[168].y
        l_third = landmarks[152].y - landmarks[1].y
        
        st.divider()
        st.header("📊 專屬骨相診斷結果")
        
        # 診斷建議
        if m_third > u_third and m_third > l_third:
            st.warning("**【特徵：知性長臉】**")
            st.write("💡 **建議：** 適合橫向暈染腮紅。減少鼻影長度，利用臥蠶增加視覺焦點，縮短中庭。")
        else:
            st.info("**【特徵：原生幼態臉】**")
            st.write("💡 **建議：** 適合強調 T 字部光澤。使用「內生光」打亮，保持面部留白的純淨感。")
            
        eye_dist = landmarks[362].x - landmarks[133].x
        if eye_dist > 0.27:
            st.warning("**【特徵：高級感寬眼距】**")
            st.write("💡 **建議：** 適合「開眼角」畫法。眼影重心向內移，山根修容稍微加深。")
        else:
            st.info("**【特徵：精緻窄眼距】**")
            st.write("💡 **建議：** 適合「狐系眼妝」。將眼影與睫毛重心向眼尾拉長。")

        st.divider()
        if st.button("🔥 獲取完整版「1對1 真人精修報告」"):
            st.balloons()
            st.write("請截圖此頁面並私訊我們的官方 LINE！")

    else:
        st.error("偵測失敗，請確保人臉清晰且無遮擋。")
