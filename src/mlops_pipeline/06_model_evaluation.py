import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def run_evaluation():
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    config_path = "config.yaml"
    if not os.path.exists(config_path):
        config_path = os.path.join("configs", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model"]
    processed_dir = config["data"]["processed_dir"]

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run 05_train_registry.py first.")
        return False

    try:
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError:
        print("[ERROR] Test data not found. Run preprocessing first.")
        return False

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n--- Model Evaluation Results ---")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():12}: {value:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("--------------------------------\n")

    os.makedirs("artifacts", exist_ok=True)

    # Save Confusion Matrix plot
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Student Depression")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Depression (0)", "Depression (1)"])
    plt.yticks(tick_marks, ["No Depression (0)", "Depression (1)"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=150)
    plt.close()

    # Save ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {metrics['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("artifacts/roc_curve.png", dpi=150)
    plt.close()

    report = {
        "stage": "MLOps Pipeline - Model Evaluation",
        "evaluated_on": "held-out test split",
        "test_rows": int(len(y_test)),
        "metrics": {name: round(float(value), 4) for name, value in metrics.items()},
        "confusion_matrix": cm.tolist(),
    }

    os.makedirs("reports", exist_ok=True)
    report_path = config["paths"]["evaluation_report"]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Evaluation report saved to {report_path}")
    print("[SUCCESS] Evaluation plots saved to artifacts/confusion_matrix.png and artifacts/roc_curve.png")
    return True


if __name__ == "__main__":
    passed = run_evaluation()
    sys.exit(0 if passed else 1)
