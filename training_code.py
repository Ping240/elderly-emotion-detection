import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ================= GPU 設定 =================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
print("GPUs:", gpus)

# ================= 路徑設定 =================
BASE_DIR = r'C:\Users\dsp523\Downloads\archive (1)'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'test')

MODEL_PATH = 'fer2013_model.h5'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 50
EPOCHS_STAGE2 = 30
NUM_CLASSES = 7

# ================= Data Generator =================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ================= MobileNetV2 Backbone =================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze backbone
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ================= Callbacks =================
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ReduceLROnPlateau(patience=4, factor=0.3, min_lr=1e-6),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# ================= Stage 1: Train classifier =================
history_stage1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks
)

# ================= Stage 2: Fine-tune last layers =================
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_stage2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks
)

# ================= Evaluation =================
model.load_weights(MODEL_PATH)
loss, acc = model.evaluate(val_generator)
print(f"\n✅ Final Accuracy: {acc * 100:.2f}%")

# ================= Plot Training Curves =================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history_stage1.history['accuracy'], label='Train Stage1')
plt.plot(history_stage1.history['val_accuracy'], label='Val Stage1')
plt.plot(history_stage2.history['accuracy'], label='Train Stage2')
plt.plot(history_stage2.history['val_accuracy'], label='Val Stage2')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_stage1.history['loss'], label='Train Stage1')
plt.plot(history_stage1.history['val_loss'], label='Val Stage1')
plt.plot(history_stage2.history['loss'], label='Train Stage2')
plt.plot(history_stage2.history['val_loss'], label='Val Stage2')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# 自動儲存成 PNG
plt.savefig('training_curves.png', dpi=300)
print("✅ Training curves saved as training_curves.png")

plt.show()
