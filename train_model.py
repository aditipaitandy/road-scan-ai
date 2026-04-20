import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ----------------------------
# Config
# ----------------------------
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS_FROZEN  = 10   # Phase 1: train head only
EPOCHS_FINETUNE = 8   # Phase 2: fine-tune top layers of base

# ----------------------------
# Data generators
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.08,
    fill_mode="nearest"
)

# Validation: only rescale, no augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class indices:", train_data.class_indices)
print(f"Training samples : {train_data.samples}")
print(f"Validation samples: {val_data.samples}")

# Save class indices so app.py can reference them
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("Class indices saved to class_indices.json")

# ----------------------------
# Build model
# ----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Phase 1: frozen

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ----------------------------
# Callbacks
# ----------------------------
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.4,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ----------------------------
# Phase 1: Train head (frozen base)
# ----------------------------
print("\n--- Phase 1: Training classification head (base frozen) ---")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FROZEN,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# ----------------------------
# Phase 2: Fine-tune top layers of base
# ----------------------------
print("\n--- Phase 2: Fine-tuning top 30 layers of MobileNetV2 ---")
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # lower LR for fine-tune
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

early_stop2 = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINETUNE,
    callbacks=[early_stop2, reduce_lr, checkpoint]
)

# ----------------------------
# Save final model
# ----------------------------
model.save("road_damage_model.h5")
print("\nFinal model saved as road_damage_model.h5")

# ----------------------------
# Save training history for review
# ----------------------------
combined_history = {
    "phase1": {k: [float(v) for v in vals] for k, vals in history1.history.items()},
    "phase2": {k: [float(v) for v in vals] for k, vals in history2.history.items()}
}
with open("training_history.json", "w") as f:
    json.dump(combined_history, f, indent=2)
print("Training history saved to training_history.json")

# ----------------------------
# Print final metrics
# ----------------------------
final_val_acc = max(history2.history.get("val_accuracy", [0]))
final_val_auc = max(history2.history.get("val_auc", [0]))
print(f"\nBest Val Accuracy : {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Best Val AUC      : {final_val_auc:.4f}")
print("\nDone. Use road_damage_model.h5 in your app.")