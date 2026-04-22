import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPooling2D, Flatten, Dropout,
    BatchNormalization, Concatenate, LSTM, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

SEED = 42
IMG_SIZE = 50
NUM_CLASSES = 2

ROOT = Path.home() / "malevis_adversarial_repro"
TRAIN_CSV = ROOT / "data" / "annotations" / "detector_annotations" / "train_detector_annotations.csv"
EVAL_CSV = ROOT / "data" / "annotations" / "detector_annotations" / "eval_detector_annotations.csv"

MODELS_DIR = ROOT / "artifacts" / "models"
LOGS_DIR = ROOT / "artifacts" / "logs"
REPORTS_DIR = ROOT / "artifacts" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def load_df(path):
    df = pd.read_csv(path)
    req = {"image_id", "isNight", "detector_label"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def one_hot_binary(y):
    return np.eye(NUM_CLASSES, dtype=np.float32)[np.array(y, dtype=int)]


def build_detector_arrays(df_part, desc="Loading detector data"):
    X_images, X_isNight, y = [], [], []

    for _, row in tqdm(df_part.iterrows(), total=len(df_part), desc=desc):
        img_path = str(row["image_id"])
        label = int(row["detector_label"])
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

    X_images = np.array(X_images, dtype="float32")
    X_isNight = np.array(X_isNight, dtype="float32").reshape(-1, 1)
    y = np.array(y, dtype=int)

    return X_images, X_isNight, y


def image_to_row_sequence(Ximg):
    n, h, w, c = Ximg.shape
    return Ximg.reshape(n, h, w * c).astype("float32")


def save_report(prefix, y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open(REPORTS_DIR / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(cm).to_csv(REPORTS_DIR / f"{prefix}_confusion_matrix.csv", index=False)

    print("\nClassification report:\n")
    print(report)
    print("\nConfusion matrix:\n")
    print(cm)


def make_val_split(X_img, X_isn, y_int, val_size=0.1):
    idx = np.arange(len(y_int))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=SEED,
        stratify=y_int,
        shuffle=True,
    )
    return (
        X_img[train_idx], X_isn[train_idx], y_int[train_idx],
        X_img[val_idx],   X_isn[val_idx],   y_int[val_idx],
    )


def train_ann_detector(X_train_img, X_train_isn, y_train_int, X_test_img, X_test_isn, y_test_int):
    prefix = "detector_ann"

    Xtr_img, Xtr_isn, ytr_int, Xval_img, Xval_isn, yval_int = make_val_split(
        X_train_img, X_train_isn, y_train_int
    )

    Xtr = np.hstack([Xtr_img.reshape(len(Xtr_img), -1), Xtr_isn])
    Xval = np.hstack([Xval_img.reshape(len(Xval_img), -1), Xval_isn])
    Xtest = np.hstack([X_test_img.reshape(len(X_test_img), -1), X_test_isn])

    ytr = one_hot_binary(ytr_int)
    yval = one_hot_binary(yval_int)
    ytest = one_hot_binary(y_test_int)

    model = Sequential([
        Input(shape=(Xtr.shape[1],)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        CSVLogger(str(LOGS_DIR / f"{prefix}_training_log.csv"), append=False),
    ]

    model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=50,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(Xtest, ytest, verbose=0)
    print(f"\n[{prefix}] Test loss: {loss:.4f} | Test acc: {acc:.4f}")

    y_pred = model.predict(Xtest, batch_size=256, verbose=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    save_report(prefix, y_test_int, y_pred_labels)
    model.save(MODELS_DIR / f"{prefix}.keras")


def train_cnn_detector(X_train_img, X_train_isn, y_train_int, X_test_img, X_test_isn, y_test_int):
    prefix = "detector_cnn"

    Xtr_img, Xtr_isn, ytr_int, Xval_img, Xval_isn, yval_int = make_val_split(
        X_train_img, X_train_isn, y_train_int
    )

    ytr = one_hot_binary(ytr_int)
    yval = one_hot_binary(yval_int)
    ytest = one_hot_binary(y_test_int)

    img_in = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = Conv2D(32, 3, activation="relu")(img_in)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, 3, activation="relu")(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, 3, activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)

    isn_in = Input(shape=(1,), name="isNight")
    h = Concatenate()([x, isn_in])
    h = Dense(128, activation="relu")(h)
    h = Dropout(0.3)(h)
    out = Dense(NUM_CLASSES, activation="softmax")(h)

    model = Model([img_in, isn_in], out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        CSVLogger(str(LOGS_DIR / f"{prefix}_training_log.csv"), append=False),
    ]

    model.fit(
        [Xtr_img, Xtr_isn], ytr,
        validation_data=([Xval_img, Xval_isn], yval),
        epochs=50,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate([X_test_img, X_test_isn], ytest, verbose=0)
    print(f"\n[{prefix}] Test loss: {loss:.4f} | Test acc: {acc:.4f}")

    y_pred = model.predict([X_test_img, X_test_isn], batch_size=256, verbose=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    save_report(prefix, y_test_int, y_pred_labels)
    model.save(MODELS_DIR / f"{prefix}.keras")


def train_rnn_detector(X_train_img, X_train_isn, y_train_int, X_test_img, X_test_isn, y_test_int):
    prefix = "detector_rnn"

    Xtr_img, Xtr_isn, ytr_int, Xval_img, Xval_isn, yval_int = make_val_split(
        X_train_img, X_train_isn, y_train_int
    )

    Xtr_seq = image_to_row_sequence(Xtr_img)
    Xval_seq = image_to_row_sequence(Xval_img)
    Xtest_seq = image_to_row_sequence(X_test_img)

    ytr = one_hot_binary(ytr_int)
    yval = one_hot_binary(yval_int)
    ytest = one_hot_binary(y_test_int)

    seq_in = Input(shape=(IMG_SIZE, IMG_SIZE * 3), name="row_sequence")
    isn_in = Input(shape=(1,), name="isNight")

    z = Bidirectional(LSTM(64, return_sequences=False))(seq_in)
    z = Dropout(0.3)(z)
    h = Concatenate()([z, isn_in])
    h = Dense(128, activation="relu")(h)
    out = Dense(NUM_CLASSES, activation="softmax")(h)

    model = Model([seq_in, isn_in], out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        CSVLogger(str(LOGS_DIR / f"{prefix}_training_log.csv"), append=False),
    ]

    model.fit(
        [Xtr_seq, Xtr_isn], ytr,
        validation_data=([Xval_seq, Xval_isn], yval),
        epochs=50,
        batch_size=64,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate([Xtest_seq, X_test_isn], ytest, verbose=0)
    print(f"\n[{prefix}] Test loss: {loss:.4f} | Test acc: {acc:.4f}")

    y_pred = model.predict([Xtest_seq, X_test_isn], batch_size=256, verbose=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    save_report(prefix, y_test_int, y_pred_labels)
    model.save(MODELS_DIR / f"{prefix}.keras")


def main():
    train_df = load_df(TRAIN_CSV)
    eval_df = load_df(EVAL_CSV)

    print("Train detector rows:", len(train_df))
    print("Eval detector rows:", len(eval_df))
    print("\nTrain detector label counts:")
    print(train_df["detector_label"].value_counts())
    print("\nEval detector label counts:")
    print(eval_df["detector_label"].value_counts())

    X_train_img, X_train_isn, y_train_int = build_detector_arrays(train_df, desc="Loading detector train")
    X_test_img, X_test_isn, y_test_int = build_detector_arrays(eval_df, desc="Loading detector eval")

    print("\nShapes:")
    print("X_train_img:", X_train_img.shape, "X_train_isn:", X_train_isn.shape, "y_train:", y_train_int.shape)
    print("X_test_img: ", X_test_img.shape,  "X_test_isn: ", X_test_isn.shape,  "y_test: ", y_test_int.shape)

    train_ann_detector(X_train_img, X_train_isn, y_train_int, X_test_img, X_test_isn, y_test_int)
    train_cnn_detector(X_train_img, X_train_isn, y_train_int, X_test_img, X_test_isn, y_test_int)
    train_rnn_detector(X_train_img, X_train_isn, y_train_int, X_test_img, X_test_isn, y_test_int)


if __name__ == "__main__":
    main()
