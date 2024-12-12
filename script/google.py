from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# 데이터셋 경로 설정
data_path = 'D:/kor_face_ai/real_t'

# 가중치 초기화 함수
def reset_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))


# GoogleNet(InceptionV3) 모델 로드
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# GoogleNet 위에 커스텀 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)  # 감정 클래스 (7개)

# 전체 모델 정의
emotion_model = Model(inputs=base_model.input, outputs=predictions)

# 일부 레이어 고정 (전이 학습)
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
emotion_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터 증강 및 로드
# Train:Validation:Test = 60%:20%:20%
datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.4  # 전체 데이터셋에서 40%를 validation 및 test로 사용
)

# Train 데이터 로드 (전체 데이터셋의 60%)
train_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # training: 60% 데이터
)

# Validation 및 Test 데이터를 위한 데이터 증강 설정
test_val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.5  # validation과 test를 나누기 위한 비율 설정 (40% 중 절반씩)
)

# Validation 데이터 로드 (전체 데이터셋의 20%)
validation_generator = test_val_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # validation: 20% 데이터
)

# Test 데이터 로드 (전체 데이터셋의 20%)
test_generator = test_val_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # test: 20% 데이터
)

# 데이터셋 개수 출력
print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# 모델 학습
history = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=100
)

# 학습된 모델 가중치 저장 (.h5 형식)
model_save_path = 'models/emotion_model/emotion_model_googlenet.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
emotion_model.save(model_save_path)

print(f'Model saved to {model_save_path}')

# 학습 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("model_accuracy.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("model_loss.png")
plt.show()

# 최종 Train, Validation 값 출력
final_train_loss = history.history['loss'][-1]
final_train_accuracy = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

print("\n최종 학습 결과:")
print(f"Train Loss: {final_train_loss:.4f}, Train Accuracy: {final_train_accuracy:.4f}")
print(f"Validation Loss: {final_val_loss:.4f}, Validation Accuracy: {final_val_accuracy:.4f}")

# 테스트 데이터 평가
test_loss, test_accuracy = emotion_model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
