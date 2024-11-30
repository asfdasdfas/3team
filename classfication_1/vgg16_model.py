# VGG-16 모델 정의 파일 (vgg16_model_normal.py)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
from tensorflow.keras import mixed_precision

# 경고 메시지 무시 (손상된 EXIF 데이터 등)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# GPU 메모리 제한 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 할당을 50%로 제한
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
    except RuntimeError as e:
        print(e)

# Mixed Precision 사용 설정
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 데이터셋 경로 (절대 경로로 수정)
DATASET_PATH = 'D:/git/img_mk/dataset/images'  # 모든 데이터가 있는 폴더 경로

# 손상된 이미지 파일 제거 함수 정의
def verify_and_remove(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()  # 기본 무결성 검사
            img = Image.open(img_path)  # 다시 열어서 검사
            img.load()  # 이미지 로드하여 문제 발생 여부 확인
            img = img.convert('RGB')  # 이미지를 RGB로 변환하여 손상 확인 강화
            if img.size[0] < 10 or img.size[1] < 10:
                raise OSError("이미지 크기가 너무 작습니다.")
            img.save(img_path, 'JPEG')  # JPEG로 다시 저장
    except (IOError, SyntaxError, ValueError, OSError) as e:
        print(f"손상된 이미지 발견 및 제거: {img_path}, 오류: {e}")
        os.remove(img_path)  # 손상된 파일 삭제

# ThreadPoolExecutor를 사용해 병렬로 손상된 이미지 제거
with ThreadPoolExecutor() as executor:
    for root, _, files in os.walk(DATASET_PATH):
        img_paths = [os.path.join(root, file) for file in files if file.endswith(('.jpg', '.jpeg'))]
        executor.map(verify_and_remove, img_paths)

# 데이터 전처리 및 증가
train_datagen = ImageDataGenerator(
    rescale=1. / 255,           # 정규화
    validation_split=0.2        # 데이터의 20%를 검증 데이터로 사용
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,           # 정규화
    validation_split=0.2        # 데이터의 20%를 테스트 데이터로 사용
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2        # 데이터의 20%를 검증 데이터로 사용
)

# 데이터셋 불러오기
def load_datasets(batch_size=8, img_size=(224, 224)):
    train_dataset = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',       # 학습용 데이터
        seed=123
    )

    validation_dataset = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',     # 검증용 데이터
        seed=123
    )

    test_dataset = test_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',     # 테스트용 데이터
        seed=123
    )

    class_names = list(train_dataset.class_indices.keys())
    return train_dataset, validation_dataset, test_dataset, class_names

# MobileNetV2 모델 정의
def build_vgg16_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')  # 클래스 수에 따라 출력 뉴럴 조정
    ])
    return model

# 데이터셋 로드
train_dataset, validation_dataset, test_dataset, class_names = load_datasets(batch_size=8)

# 클래스 수 확인
num_classes = len(class_names)
print("클래스 수:", num_classes)

# 모델 생성
model = build_vgg16_model(num_classes)

# 모델 컴파일 (기본 설정)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=30)

# 테스트 데이터셋 평가
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

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
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
