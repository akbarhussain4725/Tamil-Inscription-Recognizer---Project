import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image, UnidentifiedImageError

# ========== PATCH to Skip Bad Images ==========
original_load_img = tf.keras.utils.load_img
def safe_load_img(path, *args, **kwargs):
    try:
        return original_load_img(path, *args, **kwargs)
    except (UnidentifiedImageError, OSError):
        print(f"⚠️ Skipping unreadable image: {path}")
        return None  # Skipped in generator
tf.keras.utils.load_img = safe_load_img

# ========== Optional Cleanup Script ==========
def clean_dataset(directory):
    total_deleted = 0
    for folder, _, files in os.walk(directory):
        for f in files:
            path = os.path.join(folder, f)
            try:
                with Image.open(path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError):
                print(f"Deleting corrupted image: {path}")
                os.remove(path)
                total_deleted += 1
    print(f"✅ Cleanup complete. {total_deleted} bad images deleted.")

# Uncomment to run once before training
# clean_dataset("dataset/train")
# clean_dataset("dataset/valid")

# ========== Settings ==========
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 10
MODEL_SAVE_PATH = "tamil_3.keras"

# ========== Data Generators ==========
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    "datasett/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    "datasett/valid",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = len(train_generator.class_indices)

# ========== CNN Model ==========

model = Sequential([
    # First Conv2D layer for grayscale image input (1 channel)
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),  # (64, 64, 1) for grayscale
    MaxPooling2D(2, 2),

    # Second Conv2D layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Third Conv2D layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    # Flattening layer to convert 2D matrix to 1D
    Flatten(),

    # Dense layer with updated size for flattened output
    Dense(4608, activation='relu'),  # Ensure size matches output of Flatten layer
    Dropout(0.3),

    # Output layer with softmax activation (number of classes should match class_names)
    Dense(len(num_classes), activation='softmax')
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ========== Callbacks ==========
early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy", verbose=1)

# ========== Training ==========
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)
