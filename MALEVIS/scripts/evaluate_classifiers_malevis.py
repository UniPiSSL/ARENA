import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tensorflow.keras.models import load_model

ROOT = Path.home() / "malevis_adversarial_repro"

CLEAN_CSV = ROOT / "data" / "annotations" / "new_annotations.csv"
ADV_DIR = ROOT / "data" / "annotations" / "adv_annotations"

MODEL_DIR = ROOT / "artifacts" / "models"
OUT_DIR = ROOT / "artifacts" / "eval_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 50
NUM_CLASSES = 26

EVAL_SETS = {
    "clean_test": CLEAN_CSV,
    "Annotations_ANN_fgsm": ADV_DIR / "Annotations_ANN_fgsm.csv",
    "Annotations_ANN_pgd": ADV_DIR / "Annotations_ANN_pgd.csv",
    "Annotations_CNN_fgsm": ADV_DIR / "Annotations_CNN_fgsm.csv",
    "Annotations_CNN_pgd": ADV_DIR / "Annotations_CNN_pgd.csv",
    "Annotations_RNN_fgsm": ADV_DIR / "Annotations_RNN_fgsm.csv",
    "Annotations_RNN_pgd": ADV_DIR / "Annotations_RNN_pgd.csv",
}


def load_df(path):
    df = pd.read_csv(path)
    req = {"image_id", "label", "isNight", "split"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[df["split"] == "test"].copy()
    df["label0"] = df["label"].astype(int) - 1
    return df


def build_arrays(df, mode):
    Ximg, Xisn, y = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {mode}"):
        img_path = str(row["image_id"])
        label = int(row["label0"])
        isn = int(row["isNight"])

        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read: {img_path}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        Ximg.append(img)
        Xisn.append(isn)
        y.append(label)

    Ximg = np.array(Ximg, dtype="float32")
    Xisn = np.array(Xisn, dtype="float32").reshape(-1, 1)
    y = np.array(y, dtype=int)
    return Ximg, Xisn, y


def image_to_row_sequence(Ximg):
    n, h, w, c = Ximg.shape
    return Ximg.reshape(n, h, w * c).astype("float32")


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def pred_ann(model, Ximg, Xisn):
    X = np.hstack([Ximg.reshape(len(Ximg), -1), Xisn])
    probs = model.predict(X, batch_size=256, verbose=0)
    return np.argmax(probs, axis=1)


def pred_cnn(model, Ximg, Xisn):
    probs = model.predict([Ximg, Xisn], batch_size=256, verbose=0)
    return np.argmax(probs, axis=1)


def pred_rnn(model, Ximg, Xisn):
    Xseq = image_to_row_sequence(Ximg)
    probs = model.predict([Xseq, Xisn], batch_size=256, verbose=0)
    return np.argmax(probs, axis=1)


def main():
    ann = load_model(MODEL_DIR / "malevis_ann_model.keras")
    cnn = load_model(MODEL_DIR / "malevis_cnn_model.keras")
    rnn = load_model(MODEL_DIR / "malevis_rnn_model.keras")

    rows = []

    for dataset_name, csv_path in EVAL_SETS.items():
        df = load_df(csv_path)
        Ximg, Xisn, y = build_arrays(df, dataset_name)

        for model_name, fn in [
            ("ANN", lambda: pred_ann(ann, Ximg, Xisn)),
            ("CNN", lambda: pred_cnn(cnn, Ximg, Xisn)),
            ("RNN", lambda: pred_rnn(rnn, Ximg, Xisn)),
        ]:
            y_pred = fn()
            row = {
                "model": model_name,
                "dataset": dataset_name,
                "n_samples": len(y),
                **metrics(y, y_pred),
            }
            rows.append(row)
            print(row)

    out_df = pd.DataFrame(rows)

    model_order = ["ANN", "CNN", "RNN"]
    dataset_order = [
        "clean_test",
        "Annotations_ANN_fgsm",
        "Annotations_ANN_pgd",
        "Annotations_CNN_fgsm",
        "Annotations_CNN_pgd",
        "Annotations_RNN_fgsm",
        "Annotations_RNN_pgd",
    ]

    out_df["model"] = pd.Categorical(out_df["model"], categories=model_order, ordered=True)
    out_df["dataset"] = pd.Categorical(out_df["dataset"], categories=dataset_order, ordered=True)
    out_df = out_df.sort_values(["model", "dataset"]).reset_index(drop=True)

    out_xlsx = OUT_DIR / "evaluation_summary_macro.xlsx"
    out_csv = OUT_DIR / "evaluation_summary_macro.csv"

    out_df.to_excel(out_xlsx, index=False)
    out_df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_xlsx}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
