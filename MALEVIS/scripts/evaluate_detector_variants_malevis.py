import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from tensorflow.keras.models import load_model

ROOT = Path.home() / "malevis_adversarial_repro"
IMG_SIZE = 50

MODELS = {
    "ANN": ROOT / "artifacts" / "models" / "detector_ann.keras",
    "CNN": ROOT / "artifacts" / "models" / "detector_cnn.keras",
    "RNN": ROOT / "artifacts" / "models" / "detector_rnn.keras",
    "SEMI_CNN": ROOT / "artifacts" / "models" / "detector_semi_cnn.keras",
    "SEMI_CNN_LIFELONG": ROOT / "artifacts" / "models" / "detector_semi_cnn_lifelong.keras",
}

DET_CSV_DIR = ROOT / "data" / "annotations" / "detector_annotations"
OUT_DIR = ROOT / "artifacts" / "final_tables_thesis_5models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_SETS = {
    "known": DET_CSV_DIR / "eval_detector_annotations.csv",
    "unknown_fgsm_rnn": DET_CSV_DIR / "unknown_adversarial_fgsm_RNN_detector_annotations.csv",
    "unknown_pgd": DET_CSV_DIR / "unknown_adversarial_PGD_detector_annotations.csv",
}

def load_detector_df(path):
    df = pd.read_csv(path)
    return df

def build_arrays(df):
    X_images, X_isNight, y = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {Path(df.attrs.get('src','dataset')).name}"):
        img_path = str(row["image_id"])
        label = int(row["detector_label"])
        is_night = int(row["isNight"])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        X_images.append(img)
        X_isNight.append(is_night)
        y.append(label)
    return (
        np.array(X_images, dtype="float32"),
        np.array(X_isNight, dtype="float32").reshape(-1, 1),
        np.array(y, dtype=int),
    )

def image_to_row_sequence(Ximg):
    n, h, w, c = Ximg.shape
    return Ximg.reshape(n, h, w * c).astype("float32")

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_samples": len(y_true),
    }

def predict_for_model(model_name, model, Ximg, Xisn):
    if model_name == "ANN":
        X = np.hstack([Ximg.reshape(len(Ximg), -1), Xisn])
        probs = model.predict(X, batch_size=256, verbose=0)
    elif model_name == "RNN":
        Xseq = image_to_row_sequence(Ximg)
        probs = model.predict([Xseq, Xisn], batch_size=256, verbose=0)
    else:
        probs = model.predict([Ximg, Xisn], batch_size=256, verbose=0)
    return np.argmax(probs, axis=1)

def main():
    loaded = {name: load_model(path) for name, path in MODELS.items()}
    rows = []

    for eval_name, csv_path in EVAL_SETS.items():
        df = load_detector_df(csv_path)
        df.attrs["src"] = str(csv_path)
        Ximg, Xisn, y = build_arrays(df)

        for model_name, model in loaded.items():
            y_pred = predict_for_model(model_name, model, Ximg, Xisn)
            metrics = compute_metrics(y, y_pred)
            rows.append({
                "eval_set": eval_name,
                "model": model_name,
                **metrics
            })

    out_df = pd.DataFrame(rows)

    known = out_df[out_df["eval_set"] == "known"][["model","n_samples","accuracy","precision","recall","f1"]]
    fgsm = out_df[out_df["eval_set"] == "unknown_fgsm_rnn"][["model","n_samples","accuracy","precision","recall","f1"]]
    pgd = out_df[out_df["eval_set"] == "unknown_pgd"][["model","n_samples","accuracy","precision","recall","f1"]]

    order = ["ANN","CNN","RNN","SEMI_CNN","SEMI_CNN_LIFELONG"]
    for df in [known, fgsm, pgd]:
        df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
        df.sort_values("model", inplace=True)

    known.to_csv(OUT_DIR / "Table_4_2_final.csv", index=False)
    fgsm.to_csv(OUT_DIR / "Table_4_3_final.csv", index=False)
    pgd.to_csv(OUT_DIR / "Table_4_4_final.csv", index=False)

    print("Saved:")
    print(OUT_DIR / "Table_4_2_final.csv")
    print(OUT_DIR / "Table_4_3_final.csv")
    print(OUT_DIR / "Table_4_4_final.csv")

if __name__ == "__main__":
    main()
