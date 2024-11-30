# googleNet_model_emotion.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import warnings

# 경고 메시지 무시 (손상된 EXIF 데이터 등)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# 데이터셋 경로 (절대 경로로 수정)  
DATASET_PATH = 'D:/kor_face_ai/real_t'  # 모든 데이터가 있는 폴더 경로

# 하이퍼파라미터 설정
BATCH_SIZE = 16
IMG_SIZE = (299, 299)
EPOCHS = 30
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

# 손상된 이미지 파일 제거 함수 정의
def verify_and_remove(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()  # 기본 무결성 검사
            img = img.convert('RGB')  # 이미지를 RGB로 변환하여 손상 확인 강화
            img.save(img_path, 'JPEG')  # JPEG로 다시 저장
    except (IOError, SyntaxError, ValueError):
        print(f"손상된 이미지 발견 및 제거: {img_path}")
        os.remove(img_path)  # 손상된 파일 삭제

# ThreadPoolExecutor를 사용해 병렬로 손상된 이미지 제거
with ThreadPoolExecutor() as executor:
    for root, _, files in os.walk(DATASET_PATH):
        img_paths = [os.path.join(root, file) for file in files if file.endswith(('.jpg', '.jpeg'))]
        executor.map(verify_and_remove, img_paths)

# 데이터셋 불러오기
def load_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    # 손상된 이미지를 무시하기 전에 class_names를 먼저 가져옵니다.
    raw_train_dataset = image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )
    class_names = raw_train_dataset.class_names  # class_names 저장

    train_dataset = raw_train_dataset.apply(tf.data.experimental.ignore_errors())
    
    validation_dataset = image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    ).apply(tf.data.experimental.ignore_errors())
    
    return train_dataset, validation_dataset, class_names

# GoogleNet (Inception-V3) 모델 정의
def build_googlenet_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(DROPOUT_RATE),
        Dense(256, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation='softmax')  # 클래스 수에 따라 출력 뉴런 조정
    ])
    return model

# 데이터셋 로드
train_dataset, validation_dataset, class_names = load_datasets(batch_size=BATCH_SIZE)

# 클래스 수 확인
num_classes = len(class_names)
print("클래스 수:", num_classes)

# 모델 생성
model = build_googlenet_model(num_classes)

# 모델 컴파일 (하이퍼파라미터 적용)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

# 최종 학습 및 검증 정확도 출력
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"최종 Training Accuracy: {final_train_accuracy:.4f}")
print(f"최종 Validation Accuracy: {final_val_accuracy:.4f}")

# 학습된 모델 저장
model_save_path = "D:/git/img_mk/models"
model.save(model_save_path)
print(f"모델이 {model_save_path}에 저장되었습니다.")

# 학습 및 검증 정확도 그래프 그리기
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# 모델 불러오기 테스트
def load_emotion_model(model_path):
    return tf.keras.models.load_model(model_path)

# 모델 불러오기
loaded_model = load_emotion_model(model_save_path)
print(f"{model_save_path}에서 모델을 성공적으로 불러왔습니다.")

# 테스트 데이터셋 평가
def evaluate_model(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

# 테스트 데이터셋 로드 및 평가
test_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
).apply(tf.data.experimental.ignore_errors())

evaluate_model(loaded_model, test_dataset)
