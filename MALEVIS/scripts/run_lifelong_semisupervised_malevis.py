import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
IMG_SIZE = 50
NUM_CLASSES = 2

ROOT = Path.home() / "malevis_adversarial_repro"
TRAIN_CSV = ROOT / "data" / "annotations" / "detector_annotations" / "train_semisupervised_detector_annotations.csv"
EVAL_KNOWN_CSV = ROOT / "data" / "annotations" / "detector_annotations" / "eval_detector_annotations.csv"
EVAL_UNKNOWN_FGSM_CSV = ROOT / "data" / "annotations" / "detector_annotations" / "unknown_adversarial_fgsm_RNN_detector_annotations.csv"
EVAL_UNKNOWN_PGD_CSV = ROOT / "data" / "annotations" / "detector_annotations" / "unknown_adversarial_PGD_detector_annotations.csv"

OUT_DIR = ROOT / "artifacts" / "eval_tables"
LOGS_DIR = ROOT / "artifacts" / "logs"
MODELS_DIR = ROOT / "artifacts" / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

LABEL_FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def load_df(path):
    df = pd.read_csv(path)
    req = {"image_id", "isNight", "detector_label"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def build_arrays(df, desc="Loading"):
    X_images, X_isNight, y = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        img_path = str(row["image_id"])
        y_label = int(row["detector_label"])
        is_night = int(row["isNight"])

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        X_images.append(img)
        X_isNight.append(is_night)
        y.append(y_label)

    X_images = np.array(X_images, dtype="float32")
    X_isNight = np.array(X_isNight, dtype="float32").reshape(-1, 1)
    y = np.array(y, dtype=int)
    return X_images, X_isNight, y


def one_hot_binary(y):
    return np.eye(NUM_CLASSES, dtype=np.float32)[y]


def build_detector_model():
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
    return model


def eval_model(model, Ximg, Xisn, y_true):
    probs = model.predict([Ximg, Xisn], batch_size=256, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return acc, p, r, f1


def main():
    train_df = load_df(TRAIN_CSV)
    known_df = load_df(EVAL_KNOWN_CSV)
    unknown_fgsm_df = load_df(EVAL_UNKNOWN_FGSM_CSV)
    unknown_pgd_df = load_df(EVAL_UNKNOWN_PGD_CSV)

    X_train_all, Xisn_train_all, y_train_all = build_arrays(train_df, desc="Loading semi/lifelong train")
    X_known, Xisn_known, y_known = build_arrays(known_df, desc="Loading known eval")
    X_ufgsm, Xisn_ufgsm, y_ufgsm = build_arrays(unknown_fgsm_df, desc="Loading unknown FGSM eval")
    X_upgd, Xisn_upgd, y_upgd = build_arrays(unknown_pgd_df, desc="Loading unknown PGD eval")

    history_rows = []

    for frac in LABEL_FRACTIONS:
        idx = np.arange(len(y_train_all))

        if frac >= 1.0:
            idx_sub = idx
        else:
            idx_sub, _ = train_test_split(
                idx,
                train_size=frac,
                random_state=SEED,
                stratify=y_train_all,
                shuffle=True,
            )

        X_sub = X_train_all[idx_sub]
        Xisn_sub = Xisn_train_all[idx_sub]
        y_sub = y_train_all[idx_sub]

        tr_idx, val_idx = train_test_split(
            np.arange(len(y_sub)),
            test_size=0.1,
            random_state=SEED,
            stratify=y_sub,
            shuffle=True,
        )

        Xtr, Xval = X_sub[tr_idx], X_sub[val_idx]
        Itr, Ival = Xisn_sub[tr_idx], Xisn_sub[val_idx]
        ytr, yval = y_sub[tr_idx], y_sub[val_idx]

        model = build_detector_model()

        model.fit(
            [Xtr, Itr], one_hot_binary(ytr),
            validation_data=([Xval, Ival], one_hot_binary(yval)),
            epochs=20,
            batch_size=64,
            shuffle=True,
            verbose=1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        )

        acc_k, p_k, r_k, f1_k = eval_model(model, X_known, Xisn_known, y_known)
        acc_f, p_f, r_f, f1_f = eval_model(model, X_ufgsm, Xisn_ufgsm, y_ufgsm)
        acc_p, p_p, r_p, f1_p = eval_model(model, X_upgd, Xisn_upgd, y_upgd)

        history_rows.append({
            "label_fraction": frac,
            "train_samples": len(y_sub),
            "known_accuracy": acc_k,
            "known_precision": p_k,
            "known_recall": r_k,
            "known_f1": f1_k,
            "unknown_fgsm_accuracy": acc_f,
            "unknown_fgsm_precision": p_f,
            "unknown_fgsm_recall": r_f,
            "unknown_fgsm_f1": f1_f,
            "unknown_pgd_accuracy": acc_p,
            "unknown_pgd_precision": p_p,
            "unknown_pgd_recall": r_p,
            "unknown_pgd_f1": f1_p,
        })

        model.save(MODELS_DIR / f"lifelong_detector_frac_{str(frac).replace('.','p')}.keras")

    hist_df = pd.DataFrame(history_rows)
    hist_path = OUT_DIR / "lifelong_history.csv"
    hist_df.to_csv(hist_path, index=False)

    semi_path = OUT_DIR / "semisupervised_summary.xlsx"
    hist_df.to_excel(semi_path, index=False)

    print(f"\nSaved: {hist_path}")
    print(f"Saved: {semi_path}")
    print(hist_df)


if __name__ == "__main__":
    main()
