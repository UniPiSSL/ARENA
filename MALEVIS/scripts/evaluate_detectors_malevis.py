import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import load_model

ROOT = Path.home() / "malevis_adversarial_repro"

DET_ANN_MODEL = ROOT / "artifacts" / "models" / "detector_ann.keras"
DET_CNN_MODEL = ROOT / "artifacts" / "models" / "detector_cnn.keras"
DET_RNN_MODEL = ROOT / "artifacts" / "models" / "detector_rnn.keras"

DET_CSV_DIR = ROOT / "data" / "annotations" / "detector_annotations"
OUT_DIR = ROOT / "artifacts" / "eval_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 50

EVAL_SETS = {
    "known": DET_CSV_DIR / "eval_detector_annotations.csv",
    "unknown_fgsm_rnn": DET_CSV_DIR / "unknown_adversarial_fgsm_RNN_detector_annotations.csv",
    "unknown_pgd": DET_CSV_DIR / "unknown_adversarial_PGD_detector_annotations.csv",
}


def load_detector_df(path):
    df = pd.read_csv(path)
    req = {"image_id", "isNight", "detector_label"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def build_arrays(df):
    X_images, X_isNight, y = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {Path(df.attrs.get('src','dataset')).name}"):
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


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def eval_ann(model, Ximg, Xisn):
    X = np.hstack([Ximg.reshape(len(Ximg), -1), Xisn])
    probs = model.predict(X, batch_size=256, verbose=0)
    return np.argmax(probs, axis=1)


def eval_cnn(model, Ximg, Xisn):
    probs = model.predict([Ximg, Xisn], batch_size=256, verbose=0)
    return np.argmax(probs, axis=1)


def eval_rnn(model, Ximg, Xisn):
    Xseq = image_to_row_sequence(Ximg)
    probs = model.predict([Xseq, Xisn], batch_size=256, verbose=0)
    return np.argmax(probs, axis=1)


def main():
    det_ann = load_model(DET_ANN_MODEL)
    det_cnn = load_model(DET_CNN_MODEL)
    det_rnn = load_model(DET_RNN_MODEL)

    rows = []

    for eval_name, csv_path in EVAL_SETS.items():
        df = load_detector_df(csv_path)
        df.attrs["src"] = str(csv_path)
        Ximg, Xisn, y = build_arrays(df)

        for model_name, pred_fn in [
            ("ANN", lambda: eval_ann(det_ann, Ximg, Xisn)),
            ("CNN", lambda: eval_cnn(det_cnn, Ximg, Xisn)),
            ("RNN", lambda: eval_rnn(det_rnn, Ximg, Xisn)),
        ]:
            y_pred = pred_fn()
            metrics = compute_metrics(y, y_pred)

            row = {
                "eval_set": eval_name,
                "detector_model": model_name,
                **metrics,
            }
            rows.append(row)

            print(f"\n[{eval_name}] [{model_name}]")
            print(row)

    out_df = pd.DataFrame(rows)

    known_df = out_df[out_df["eval_set"] == "known"].copy()
    unknown_fgsm_df = out_df[out_df["eval_set"] == "unknown_fgsm_rnn"].copy()
    unknown_pgd_df = out_df[out_df["eval_set"] == "unknown_pgd"].copy()

    known_path = OUT_DIR / "evaluation_detector_summary.xlsx"
    fgsm_path = OUT_DIR / "evaluation_detector_unknown_fgsm_RNN_summary.xlsx"
    pgd_path = OUT_DIR / "evaluation_detector_unknown_PGD_summary.xlsx"

    known_df.to_excel(known_path, index=False)
    unknown_fgsm_df.to_excel(fgsm_path, index=False)
    unknown_pgd_df.to_excel(pgd_path, index=False)

    print(f"\nSaved: {known_path}")
    print(f"Saved: {fgsm_path}")
    print(f"Saved: {pgd_path}")


if __name__ == "__main__":
    main()
