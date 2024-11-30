from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 데이터셋 경로 설정
data_path = 'D:/git/img_mk/dataset/images'

# GoogleNet(InceptionV3) 모델 로드 (사전 학습된 가중치 사용)
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# GoogleNet 모델 위에 감정 분류를 위한 커스텀 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)  # 글로벌 평균 풀링
x = Dense(128, activation='relu')(x)  # 완전 연결층
predictions = Dense(7, activation='softmax')(x)  # 감정 클래스 (7개)

# 전체 모델 정의
emotion_model = Model(inputs=base_model.input, outputs=predictions)

# 일부 레이어 고정 (전이 학습)
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
emotion_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터 증강 및 로드 (train, validation, test 비율 나누기)
datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 전체 데이터셋에서 40%를 validation 및 test로 사용 (6:2:2 비율을 위해)
)

# Train 데이터 로드 (전체 데이터셋에서 60% 사용)
train_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # train 데이터 (60%)
)


# Validation 데이터 로드 (전체 데이터셋에서 20% 사용)
validation_datagen = ImageDataGenerator(
    rescale=1.0/255,
    #validation_split=0.5  # validation 및 test로 나눈 40% 중에서 절반 사용 (즉, 20%) test진행 하지 않으면 여기 #처리 하셈
)

validation_generator = validation_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # validation 데이터 (20%)
)


# Test 데이터 로드 (전체 데이터셋에서 20% 사용)
test_generator = validation_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # test 데이터 (20%)
)


# 모델 학습
emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# 학습된 모델 가중치 저장 (.h5 형식)
model_save_path = 'models/emotion_model/emotion_model.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
emotion_model.save(model_save_path)

print(f'Model saved to {model_save_path}')


# 모델 평가 (테스트 데이터셋)
test_loss, test_accuracy = emotion_model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test Accuracy: {test_accuracy:.2f}')
