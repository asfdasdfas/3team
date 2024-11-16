import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import warnings

# 경고 메시지 무시 (손상된 EXIF 데이터 등)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# 데이터셋 경로 (절대 경로로 수정)
DATASET_PATH = 'D:/kor_face_ai/real_t'  # 모든 데이터가 있는 폴더 경로

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
def load_datasets(batch_size=32, img_size=(32, 32)):  # LeNet-5는 32x32 입력 사용
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

# LeNet-5 모델 정의
def build_lenet5_model(num_classes):
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 3)),
        AveragePooling2D(),
        Conv2D(16, kernel_size=(5, 5), activation='tanh'),
        AveragePooling2D(),
        Flatten(),
        Dense(120, activation='tanh'),
        Dense(84, activation='tanh'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# 하이퍼파라미터 설정
BATCH_SIZE = 16
IMG_SIZE = (32, 32)  # LeNet-5는 32x32 입력 크기를 사용

# 데이터셋 로드
train_dataset, validation_dataset, class_names = load_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# 클래스 수 확인
num_classes = len(class_names)
print("클래스 수:", num_classes)

# 모델 생성
model = build_lenet5_model(num_classes)

# 모델 컴파일 (학습률 조정)
model.compile(optimizer=Adam(learning_rate=0.0002),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=50)

# 최종 학습 및 검증 정확도 출력
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history.get('val_accuracy', [None])[-1]
if final_val_accuracy is not None:
    print(f"최종 Training Accuracy: {final_train_accuracy:.4f}")
    print(f"최종 Validation Accuracy: {final_val_accuracy:.4f}")
else:
    print(f"최종 Training Accuracy: {final_train_accuracy:.4f}")
    print("Validation Accuracy가 기록되지 않았습니다.")


# 학습 및 검증 정확도 그래프 그리기
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
