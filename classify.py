import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time
from collections import Counter

# 데이터셋 클래스 정의
class_to_emotion = {
    'angry': 0, 
    'Anxiety': 1, 
    'happy': 2,
    'neutrality': 3, 
    'Panic': 4,
    'sad': 5,
    'Wound': 6
}

# 데이터 로드 함수 정의
def load_data_from_dir(data_dir):
    images, labels = [], []

    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        if os.path.isdir(label_path):
            label = class_to_emotion.get(label_name, -1)
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                if image_file.endswith('.jpg'):
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (96, 96))
                        img = img / 255.0  # 정규화
                        images.append(img)
                        labels.append(label)
    return np.array(images), np.array(labels)

# 데이터 로드
train_x, train_y = load_data_from_dir('./face/train/cas2')
valid_x, valid_y = load_data_from_dir('./face/val/cas2')
test_x, test_y = load_data_from_dir('C:/Users/KNUT/face/test/cas')

# 데이터 형식 변환
train_x = train_x.reshape(train_x.shape[0], 96, 96, 1)
valid_x = valid_x.reshape(valid_x.shape[0], 96, 96, 1)
test_x = test_x.reshape(test_x.shape[0], 96, 96, 1)

# 원-핫 인코딩
train_y = tf.keras.utils.to_categorical(train_y, 7)
valid_y = tf.keras.utils.to_categorical(valid_y, 7)
test_y = tf.keras.utils.to_categorical(test_y, 7)

# 클래스 불균형 확인
print("Train data distribution:", Counter(train_y.argmax(axis=1)))
print("Validation data distribution:", Counter(valid_y.argmax(axis=1)))
print("Test data distribution:", Counter(test_y.argmax(axis=1)))

# ResNet50 모델 정의
def ResNet50(classes):
    X_input = keras.layers.Input(shape=[96, 96, 1])

    X = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_input)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.ReLU()(X)
    X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    def bottleneck_residual_block(X, filters, reduce=False, s=2):
        F1, F2, F3 = filters
        X_shortcut = X

        if reduce:
            X = keras.layers.Conv2D(F1, (1, 1), strides=(s, s), kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
            X = keras.layers.BatchNormalization(axis=3)(X)
            X = keras.layers.ReLU()(X)

            X_shortcut = keras.layers.Conv2D(F3, (1, 1), strides=(s, s), kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_shortcut)
            X_shortcut = keras.layers.BatchNormalization(axis=3)(X_shortcut)
        else:
            X = keras.layers.Conv2D(F1, (1, 1), strides=(1, 1), kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
            X = keras.layers.BatchNormalization(axis=3)(X)
            X = keras.layers.ReLU()(X)

        X = keras.layers.Conv2D(F2, (3, 3), strides=(1, 1), padding='same', kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = keras.layers.BatchNormalization(axis=3)(X)
        X = keras.layers.ReLU()(X)

        X = keras.layers.Conv2D(F3, (1, 1), strides=(1, 1), kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = keras.layers.BatchNormalization(axis=3)(X)

        X = keras.layers.Add()([X, X_shortcut])
        X = keras.layers.ReLU()(X)
        return X

    X = bottleneck_residual_block(X, [64, 64, 256], reduce=True, s=1)
    for _ in range(2):
        X = bottleneck_residual_block(X, [64, 64, 256])

    X = bottleneck_residual_block(X, [128, 128, 512], reduce=True)
    for _ in range(3):
        X = bottleneck_residual_block(X, [128, 128, 512])

    X = bottleneck_residual_block(X, [256, 256, 1024], reduce=True)
    for _ in range(5):
        X = bottleneck_residual_block(X, [256, 256, 1024])

    X = bottleneck_residual_block(X, [512, 512, 2048], reduce=True)
    for _ in range(2):
        X = bottleneck_residual_block(X, [512, 512, 2048])

    X = keras.layers.AveragePooling2D((1, 1))(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(512, activation='relu')(X)
    X = keras.layers.Dropout(0.5)(X)
    X = keras.layers.Dense(classes, activation='softmax')(X)

    model = keras.models.Model(inputs=X_input, outputs=X)
    return model

# 모델 생성 및 컴파일
model = ResNet50(7)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
]

# 학습
start_time = time.time()
history = model.fit(
    train_x, train_y,
    validation_data=(valid_x, valid_y),
    epochs=30, batch_size=16, callbacks=callbacks
)
print(f"학습 시간: {time.time() - start_time:.2f}초")

# 학습 결과 시각화
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# 모델 평가
model = keras.models.load_model('model.h5')
test_loss, test_acc = model.evaluate(test_x, test_y)
print(f"테스트 정확도: {test_acc:.4f}")

# 예측 및 혼동 행렬
pred = model.predict(test_x)
y_true = test_y.argmax(axis=1)
y_pred = pred.argmax(axis=1)

print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))
