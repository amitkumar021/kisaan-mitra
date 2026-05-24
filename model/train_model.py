# model/train_model.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json, os, matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

# ── YOUR EXACT DATASET PATH ──────────────────────────────
DATASET_PATH = "/Users/macbook/Downloads/archive (1)/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH   = os.path.join(DATASET_PATH, "valid")

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 15
# ─────────────────────────────────────────────────────────

# Verify paths exist
print("\nChecking paths...")
if not os.path.exists(TRAIN_PATH):
    print("ERROR: Train folder not found:", TRAIN_PATH)
    exit()
if not os.path.exists(VAL_PATH):
    print("ERROR: Valid folder not found:", VAL_PATH)
    exit()

print("Train path found ✓")
print("Val path found ✓")
print("Number of classes:", len(os.listdir(TRAIN_PATH)))

# Data generators
print("\nLoading images...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"Classes found: {NUM_CLASSES}")
print(f"Training images: {train_gen.samples}")
print(f"Validation images: {val_gen.samples}")

# Save class names
class_names = {str(v): k for k, v in train_gen.class_indices.items()}
with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
print("Saved class_names.json ✓")

# Build model
print("\nBuilding model...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Model built ✓")

# Phase 1 — train top layers only
print("\n--- Phase 1: Training (this will take 1-2 hours on Intel Mac) ---")
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
    ModelCheckpoint("plant_model.h5", save_best_only=True, verbose=1)
]

history1 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# Phase 2 — fine tuning
print("\n--- Phase 2: Fine-tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks
)

# Final evaluation
print("\n--- Final Results ---")
loss, acc = model.evaluate(val_gen)
print(f"Final Accuracy: {acc * 100:.2f}%")

# Save accuracy graph
all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val = history1.history['val_accuracy'] + history2.history['val_accuracy']
plt.figure(figsize=(10, 4))
plt.plot(all_acc, label='Train accuracy')
plt.plot(all_val, label='Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("training_graph.png")
plt.show()
print("Graph saved ✓")

print("\n✓ Training complete!")
print("Files saved in model/ folder:")
print("  - plant_model.h5")
print("  - class_names.json")
print("  - training_graph.png")