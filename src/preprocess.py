import os
import sys
import json
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils import (
    DATASET_NAME, NUMERIC_FEATURES, CATEGORICAL_FEATURES, load_and_split, ensure_dirs
)


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")
    ensure_dirs()

    # 1-3. Load raw data, clean it (median-fill age/CGPA) and split (stratified)
    X_train, X_test, y_train, y_test = load_and_split(impute_missing=True)

    # 4. Column types (explicit lists defined in src/utils.py)
    num_cols = NUMERIC_FEATURES
    cat_cols = CATEGORICAL_FEATURES

    # 5. Scale & Encode
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])

    # Combine features
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    # 6. Save artifacts with explicit integer type casting
    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.int64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.int64))

    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(ohe, "models/ohe.pkl")

    # Save metadata
    metadata = {
        "dataset_name": DATASET_NAME,
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols),
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
