# -*- coding: utf-8 -*-
"""Untitled15.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pGMtOSIK72c5qD551K61TAaFrXKmW8hm
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from google.colab import files
import matplotlib.pyplot as plt

# 모델 로드
model = load_model('model.h5')

# 클래스 라벨 정의 (예: stop, left, right)
class_labels = ['stop', 'left', 'right']  # 4번째 클래스는 제외

# 이미지 전처리 함수
def preprocess_image(img):
    img = img.resize((64, 64))  # 모델에 맞는 크기로 리사이즈 (64x64)
    img_array = np.array(img) / 255.0  # 이미지 정규화
    if img_array.shape[-1] == 4:  # RGBA인 경우 RGB로 변환
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    return img_array

# 이미지 업로드 및 예측
uploaded = files.upload()

for fn in uploaded.keys():
    # 업로드된 이미지 열기
    img = Image.open(io.BytesIO(uploaded[fn]))
    img.show()  # 업로드된 이미지 확인

    # 이미지 전처리
    img_array = preprocess_image(img)

    # 예측
    prediction = model.predict(img_array)
    prediction = np.squeeze(prediction)  # 1D 배열로 변환

    # 가장 높은 확률의 클래스 선택
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]  # 해당 인덱스에 해당하는 클래스 라벨

    # 결과 출력
    confidence = np.max(prediction) * 100  # 확률을 백분율로 표시
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    # 예측 확률 그래프 (4번째 클래스는 제외)
    prediction = prediction[:3]  # 4번째 클래스 제외
    plt.figure(figsize=(6, 4))
    plt.bar(class_labels, prediction, color='skyblue')
    plt.xlabel('Traffic Sign Class')
    plt.ylabel('Prediction Confidence')
    plt.title(f'Prediction Confidence for {predicted_class}')
    plt.ylim(0, 1)
    plt.show()