# 필요한 라이브러리 임포트
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 데이터셋 로드 및 전처리
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# 1. VGG-16 기반의 모델 정의
def build_vgg16_model():
    base_model = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# 2. LeNet-5 기반의 모델 정의
def build_lenet5_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    return model

# 3. GoogleNet (Inception-V3) 기반의 모델 정의
def build_googlenet_model():
    base_model = InceptionV3(weights=None, include_top=False, input_shape=(32, 32, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# 모델 빌드 및 컴파일
vgg16_model = build_vgg16_model()
lenet5_model = build_lenet5_model()
googlenet_model = build_googlenet_model()

models = [vgg16_model, lenet5_model, googlenet_model]

for model in models:
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

# 각 모델 학습
for i, model in enumerate(models):
    print(f"\nTraining Model {i+1}...\n")
    model.fit(train_images, train_labels, epochs=30, validation_data=(test_images, test_labels))
