import os
import sys
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import (
    DATASET_NAME, NUMERIC_FEATURES, CATEGORICAL_FEATURES, load_and_split, ensure_dirs
)


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")
    ensure_dirs()

    # 1-3. Load, clean and split. Missing age/CGPA stay as NaN here: the pipeline
    # imputer learns the median from the TRAINING rows only (prevents leakage).
    X_train, X_test, y_train, y_test = load_and_split(impute_missing=False)

    # 4. Column types
    num_cols = list(NUMERIC_FEATURES)
    cat_cols = list(CATEGORICAL_FEATURES)

    # 5. Modular scikit-learn pipelines
    # Numerical: fill missing with the training median, then scale
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical: one-hot encode
    cat_pipeline = Pipeline(steps=[
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ]
    )

    # 6. Execute pipeline
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    # 7. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    # Single preprocessor artifact instead of separate scaler/ohe
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    metadata = {
        "dataset_name": DATASET_NAME,
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
