import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

# =========================
# Config
# =========================
SEED = 42
IMG_SIZE = 50
NUM_CLASSES = 26

ROOT = Path.home() / "malevis_adversarial_repro"
CSV_PATH = ROOT / "data" / "annotations" / "new_annotations.csv"
MODELS_DIR = ROOT / "artifacts" / "models"
LOGS_DIR = ROOT / "artifacts" / "logs"
REPORTS_DIR = ROOT / "artifacts" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "malevis_ann_model.keras"
TRAIN_LOG_PATH = LOGS_DIR / "ann_training_log.csv"
CLASS_REPORT_PATH = REPORTS_DIR / "ann_test_classification_report.txt"
CONF_MATRIX_PATH = REPORTS_DIR / "ann_test_confusion_matrix.csv"

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# =========================
# Load annotations
# =========================
df = pd.read_csv(CSV_PATH)

required_cols = {"image_id", "label", "isNight", "split"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# map labels 1..26 -> 0..25
df["label"] = df["label"].astype(int) - 1
assert df["label"].between(0, NUM_CLASSES - 1).all(), "Labels outside 0..25"

print("Split counts:")
print(df["split"].value_counts())
print("\nLabels per split:")
print(pd.crosstab(df["split"], df["label"]))

# =========================
# Data builder
# =========================
def build_Xy(df_part):
    X_images, X_isNight, y = [], [], []

    for _, row in tqdm(df_part.iterrows(), total=len(df_part), desc=f"Loading {df_part['split'].iloc[0]}"):
        img_path = str(row["image_id"])
        label = int(row["label"])
        is_night = int(row["isNight"])

        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0

        X_images.append(img)
        X_isNight.append(is_night)
        y.append(label)

    X_images = np.array(X_images).reshape(len(X_images), -1)
    X_isNight = np.array(X_isNight).reshape(-1, 1)
    X_combined = np.hstack([X_images, X_isNight])

    y = to_categorical(np.array(y), num_classes=NUM_CLASSES)
    return X_combined, y

# =========================
# Build splits
# =========================
X_train, y_train = build_Xy(df[df["split"] == "train"].copy())
X_val, y_val = build_Xy(df[df["split"] == "val"].copy())
X_test, y_test = build_Xy(df[df["split"] == "test"].copy())

print("\nShapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape,  "y_test: ", y_test.shape)

# =========================
# Model
# =========================
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Train
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

csv_logger = CSVLogger(str(TRAIN_LOG_PATH), append=False)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    shuffle=True,
    callbacks=[early_stop, csv_logger],
    verbose=1
)

# =========================
# Evaluate
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

y_pred = model.predict(X_test, batch_size=256, verbose=1)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

report = classification_report(y_true_labels, y_pred_labels, digits=4)
cm = confusion_matrix(y_true_labels, y_pred_labels)

print("\nClassification report:\n")
print(report)
print("\nConfusion matrix:\n")
print(cm)

with open(CLASS_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

pd.DataFrame(cm).to_csv(CONF_MATRIX_PATH, index=False)

# =========================
# Save model
# =========================
model.save(MODEL_PATH)
print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved training log to: {TRAIN_LOG_PATH}")
print(f"Saved classification report to: {CLASS_REPORT_PATH}")
print(f"Saved confusion matrix to: {CONF_MATRIX_PATH}")
