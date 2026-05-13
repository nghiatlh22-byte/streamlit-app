import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Cấu hình trang
st.set_page_config(
    page_title="Nhận Diện Cảm Xúc Khuôn Mặt",
    page_icon="😶",
    layout="centered"
)

# Tiêu đề
st.title("😶 Nhận Diện Cảm Xúc Khuôn Mặt")
st.markdown("**Mô hình CNN - FER2013 Dataset**")

# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_emotion_model.keras')
        st.success("✅ Đã tải mô hình thành công!")
        return model
    except:
        st.error("❌ Không tìm thấy file mô hình 'best_emotion_model.keras'")
        st.info("Bạn cần upload file mô hình vào repository")
        return None

model = load_model()

# Danh sách cảm xúc
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_vn = ['Giận dữ', 'Ghê tởm', 'Sợ hãi', 'Vui vẻ', 'Bình thường', 'Buồn', 'Ngạc nhiên']

# Upload ảnh
st.subheader("📤 Tải ảnh lên để dự đoán")
uploaded_file = st.file_uploader("Chọn ảnh khuôn mặt (jpg, jpeg, png)", 
                                type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file).convert('L')  # Chuyển sang grayscale
    st.image(image, caption="Ảnh đã tải lên", width=300)
    
    # Tiền xử lý
    img_resized = image.resize((48, 48))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)   # Thêm kênh
    img_array = np.expand_dims(img_array, axis=0)    # Thêm batch

    # Dự đoán
    with st.spinner("Đang dự đoán..."):
        predictions = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index] * 100

    # Hiển thị kết quả
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Cảm xúc:** {emotions[predicted_index]}")
        st.success(f"**Tiếng Việt:** {emotion_vn[predicted_index]}")
    with col2:
        st.info(f"**Độ tự tin:** {confidence:.2f}%")

    # Biểu đồ xác suất
    st.subheader("📊 Xác suất các cảm xúc")
    prob_df = {emotion: float(prob*100) for emotion, prob in zip(emotions, predictions[0])}
    st.bar_chart(prob_df)

else:
    st.info("👆 Vui lòng tải ảnh lên để bắt đầu dự đoán")

# Thông tin thêm
with st.expander("ℹ️ Thông tin về mô hình"):
    st.write("""
    - Dataset: FER2013 (Kaggle)
    - Input size: 48x48 grayscale
    - Số lớp: 7 cảm xúc
    - Framework: TensorFlow + Keras
    """)

st.caption("Made with ❤️ using Streamlit & TensorFlow")
