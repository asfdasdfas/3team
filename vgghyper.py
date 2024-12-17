import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 하이퍼파라미터 설정
img_size = 224  # VGG16이 요구하는 입력 크기
batch_size = 64  # 배치 크기 64로 변경
num_classes = 7  # 표정 클래스 수
epochs = 30  # 에포크 수 50으로 증가

# 데이터 준비: ImageDataGenerator를 이용해 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,  # 회전 범위 증가
    width_shift_range=0.3,  # 수평 이동 범위 증가
    height_shift_range=0.3,  # 수직 이동 범위 증가
    shear_range=0.3,  # 기울기 범위 증가
    zoom_range=0.3,  # 확대 범위 증가
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]  # 밝기 범위 추가
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_data_dir = "./face/train/cas"  # 훈련 데이터 디렉토리
val_data_dir = "./face/val/cas"      # 검증 데이터 디렉토리

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# VGG16 모델 불러오기 (pre-trained weights 사용)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# 상단 분류층 추가
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)  # 노드 수 증가
x = Dropout(0.4)(x)  # 드롭아웃 비율을 0.4로 설정
predictions = Dense(num_classes, activation='softmax')(x)

# 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# VGG16의 기존 가중치는 고정 (특징 추출기 역할)
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일 (학습률 조정)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 조기 종료 및 학습률 감소 콜백 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# 모델 학습
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# 학습 성능 시각화 함수
def plot_accuracy_and_loss(history, title="Model Performance"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # 정확도
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    
    # 손실
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    
    plt.suptitle(title, fontsize=16)
    plt.show()

# 초기 학습 성능 시각화
plot_accuracy_and_loss(history, title="Training Performance with Adjusted Hyperparameters")

# 최종 훈련 및 검증 정확도 출력
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {train_acc * 100:.2f}%")
print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

# 모델 저장
model.save("vgg16_emotion_classifier_improved.h5")
