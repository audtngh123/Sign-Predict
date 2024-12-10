import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# 모델 로드
model = load_model('model.h5')

# 클래스 라벨 정의
class_labels = ['stop', 'left', 'right']  # 4번째 클래스는 제외

# 이미지 전처리 함수
def preprocess_image(img):
    img = img.resize((64, 64))  # 모델에 맞는 크기로 리사이즈 (64x64)
    img_array = np.array(img) / 255.0  # 이미지 정규화
    if img_array.shape[-1] == 4:  # RGBA인 경우 RGB로 변환
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# Streamlit 앱 구성
st.title("Traffic Sign Classification")
st.write("Upload an image to classify the traffic sign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 이미지 로드
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 이미지 전처리
    img_array = preprocess_image(img)

    # 예측 수행
    prediction = model.predict(img_array)
    prediction = np.squeeze(prediction)  # 1D 배열로 변환

    # 가장 높은 확률의 클래스 선택
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]  # 해당 인덱스에 해당하는 클래스 라벨

    # 결과 출력
    confidence = np.max(prediction) * 100  # 확률을 백분율로 표시
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # 예측 확률 그래프 (4번째 클래스는 제외)
    prediction = prediction[:3]  # 4번째 클래스 제외
    fig, ax = plt.subplots()
    ax.bar(class_labels, prediction, color='skyblue')
    ax.set_xlabel('Traffic Sign Class')
    ax.set_ylabel('Prediction Confidence')
    ax.set_title(f'Prediction Confidence for {predicted_class}')
    ax.set_ylim(0, 1)

    st.pyplot(fig)