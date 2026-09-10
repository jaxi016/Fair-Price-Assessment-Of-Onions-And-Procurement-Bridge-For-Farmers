import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ============================================================
# Configuration
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)

CLASSES = ["Grade A", "Grade B", "Grade C"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ============================================================
# Collect images
# ============================================================

def collect_images():
    items = []

    for label, class_name in enumerate(CLASSES):
        folder = DATA / class_name

        files = [
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for p in files:
            items.append((str(p), label))

    return items


items = collect_images()

counts = {
    class_name: sum(y == i for _, y in items)
    for i, class_name in enumerate(CLASSES)
}

print("\nDataset:")
for class_name, count in counts.items():
    print(f"  {class_name}: {count}")

if any(count == 0 for count in counts.values()):
    raise SystemExit(
        f"All three classes need images. Counts = {counts}"
    )

# ============================================================
# Split
#
# 70% train
# 15% validation
# 15% test
# ============================================================

paths = np.array([p for p, _ in items])
labels = np.array([y for _, y in items])

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SEED,
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SEED,
)

print("\nSplit:")
print("  Training:   ", len(train_paths))
print("  Validation: ", len(val_paths))
print("  Test:       ", len(test_paths))

# ============================================================
# Dataset loader
# ============================================================

def load_image(path, label):
    data = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        data,
        channels=3,
        try_recover_truncated=True,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    image = tf.cast(image, tf.float32)

    return image, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(
    train_paths,
    train_labels,
    training=True,
)

val_ds = make_dataset(
    val_paths,
    val_labels,
)

test_ds = make_dataset(
    test_paths,
    test_labels,
)

# ============================================================
# Class weights
# ============================================================

class_counts = np.bincount(
    train_labels,
    minlength=len(CLASSES),
)

total = len(train_labels)

class_weights = {}

for i, count in enumerate(class_counts):
    class_weights[i] = total / (
        len(CLASSES) * count
    )

print("\nClass weights:")
for i, class_name in enumerate(CLASSES):
    print(
        f"  {class_name}: "
        f"{class_weights[i]:.4f}"
    )

# ============================================================
# Data augmentation
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),
        tf.keras.layers.RandomRotation(
            0.08
        ),
        tf.keras.layers.RandomZoom(
            0.12
        ),
        tf.keras.layers.RandomContrast(
            0.12
        ),
    ],
    name="augmentation",
)

# ============================================================
# EfficientNetB0 transfer learning
# ============================================================

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    ),
)

base.trainable = False

inputs = tf.keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    )
)

x = augmentation(inputs)

x = base(
    x,
    training=False,
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)

outputs = tf.keras.layers.Dense(
    len(CLASSES),
    activation="softmax",
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
)

# ============================================================
# Callbacks
# ============================================================

checkpoint_path = OUT / "best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ============================================================
# Stage 1: train classification head
# ============================================================

print("\n==========================================")
print("Stage 1: Transfer learning")
print("==========================================")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ============================================================
# Stage 2: fine-tune final EfficientNet layers
# ============================================================

print("\n==========================================")
print("Stage 2: Fine tuning")
print("==========================================")

base.trainable = True

# Freeze earlier layers.
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=callbacks,
)

# ============================================================
# Load best model
# ============================================================

best_model = tf.keras.models.load_model(
    checkpoint_path
)

# ============================================================
# Test predictions
# ============================================================

print("\n==========================================")
print("Final test evaluation")
print("==========================================")

probabilities = best_model.predict(
    test_ds,
    verbose=1,
)

predictions = np.argmax(
    probabilities,
    axis=1,
)

# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    test_labels,
    predictions,
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
)

cm = confusion_matrix(
    test_labels,
    predictions,
)

report = classification_report(
    test_labels,
    predictions,
    target_names=CLASSES,
    zero_division=0,
)

print("\nTest accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nWeighted precision:")
print(f"{precision * 100:.2f}%")

print("\nWeighted recall:")
print(f"{recall * 100:.2f}%")

print("\nWeighted F1:")
print(f"{f1 * 100:.2f}%")

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(report)

# ============================================================
# Save metrics
# ============================================================

metrics = {
    "test_accuracy": float(accuracy),
    "test_accuracy_percent": float(
        accuracy * 100
    ),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1": float(f1),
    "confusion_matrix": cm.tolist(),
    "classes": CLASSES,
    "dataset_counts": counts,
    "train_samples": int(len(train_paths)),
    "validation_samples": int(len(val_paths)),
    "test_samples": int(len(test_paths)),
    "class_weights": {
        CLASSES[i]: float(class_weights[i])
        for i in range(len(CLASSES))
    },
    "warning": (
        "Grade B is synthetic in this development run. "
        "This test accuracy must NOT be presented as "
        "final real-world Grade-B accuracy."
    ),
}

metrics_path = OUT / "model_metrics.json"

metrics_path.write_text(
    json.dumps(
        metrics,
        indent=2
    ),
    encoding="utf-8",
)

print("\nMetrics saved to:")
print(metrics_path)

print("\nTraining complete.")