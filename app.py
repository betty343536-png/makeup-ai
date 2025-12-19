import streamlit as st
import numpy as np
from PIL import Image

# 1. 高級感精品視覺設定 (移除藥丸、加入大理石質感)
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
    .stSuccess { background-color: #FFF; border: 1px solid #D4AF37; color: #5D4037; }
    </style>
    """, unsafe_allow_html=True)

# 2. 核心 AI 引擎加載 (針對 Python 3.11 優化)
@st.cache_resource
def get_ai_engine():
    try:
        import mediapipe as mp
        # 兼容多種加載路徑
        try:
            mp_fm = mp.solutions.face_mesh
        except:
            import mediapipe.python.solutions.face_mesh as mp_fm
            
        engine = mp_fm.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1,
            refine_landmarks=True
        )
        return engine, mp_fm
    except Exception as e:
        return None, None

# 3. 網頁介面呈現
st.title("AI 骨相美學診斷室")
st.write("Aesthetic Facial Proportions Analysis")

face_mesh, mp_fm = get_ai_engine()

# 檢查引擎狀態
if face_mesh is None:
    st.info("系統環境優化中，請稍候 30 秒並點擊重新整理。")
    if st.button("手動重新整理頁面"):
        st.rerun()
else:
    st.write("---")
    # 上傳功能
    uploaded_file = st.file_uploader("請拍攝或選取一張正面素顏照", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # 執行 AI 偵測
        results = face_mesh.process(img_array)
        
        if results.multi_face_landmarks:
            st.success("✅ 面部數據偵測成功！")
            st.image(image, caption="已讀取面部比例數據", use_container_width=True)
            
            # 這裡可以放簡單的分析邏輯
            landmarks = results.multi_face_landmarks[0].landmark
            # 簡單計算中庭比例 (示意)
            m_third = landmarks[1].y - landmarks[168].y
            
            st.divider()
            st.subheader("📊 初步骨相分析報告")
            
            if m_third > 0.2: # 舉例數值
                st.write("💡 **特徵：** 知性長臉感。")
                st.write("💡 **妝容建議：** 適合橫向腮紅，縮短視覺中庭。")
            else:
                st.write("💡 **特徵：** 原生幼態臉。")
                st.write("💡 **妝容建議：** 適合清透底妝，保持面部留白。")
                
            st.divider()
            if st.button("🔥 獲取完整版「1對1 真人精修報告」"):
                st.balloons()
                st.write("請截圖此頁面，並私訊預約您的專業美容師。")
        else:
            st.error("未能辨識臉部，請確保照片光線充足且無遮擋。")
