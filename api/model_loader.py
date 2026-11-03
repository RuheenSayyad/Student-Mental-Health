from pathlib import Path
from typing import Any
import json
import joblib
import yaml


class ModelLoader:
    """
    Loads the ML model and preprocessing pipeline from the
    verified MLflow deployment bundle.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path

        self.config: dict[str, Any] = {}
        self.model = None
        self.preprocessor = None

        self.model_uri = None
        self.model_name = None
        self.model_version = None
        self.run_id = None
        self.metadata = {}

    def load_config(self) -> None:
        if not self.config_path.exists():
            alt_path = self.project_root / "configs" / "config.yaml"
            if alt_path.exists():
                self.config_path = alt_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

    def load_deployment_bundle(self) -> None:
        deployment_dir = self.project_root / "deployment"
        model_path = deployment_dir / "model.pkl"
        preprocessor_path = deployment_dir / "preprocessor.pkl"
        metadata_path = deployment_dir / "model_metadata.json"

        # Fallback to models/ if deployment/ is not yet populated
        if not model_path.exists():
            fallback_model = self.project_root / "models" / "student_model.pkl"
            if not fallback_model.exists():
                fallback_model = self.project_root / "models" / "random_forest_model.pkl"
            if fallback_model.exists():
                model_path = fallback_model

        if not preprocessor_path.exists():
            fallback_pre = self.project_root / "models" / "preprocessor.pkl"
            if fallback_pre.exists():
                preprocessor_path = fallback_pre

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as file:
                self.metadata = json.load(file)
            self.model_name = self.metadata.get("model_name", "Student_Depression_Production_Model")
            self.model_version = self.metadata.get("model_version", "1")
            self.run_id = self.metadata.get("run_id", "local")
        else:
            self.model_name = "Student_Depression_Production_Model"
            self.model_version = "1"
            self.run_id = "local"

        self.model_uri = str(model_path)
        print(f"[INFO] Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print("[SUCCESS] Model loaded successfully.")

        print(f"[INFO] Loading preprocessor from: {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)
        print("[SUCCESS] Preprocessor loaded successfully.")

    def load(self) -> None:
        print("[INFO] Initializing ModelLoader...")
        self.load_config()
        self.load_deployment_bundle()
        print("[SUCCESS] ModelLoader initialization completed.")

    def is_ready(self) -> bool:
        return self.model is not None and self.preprocessor is not None


if __name__ == "__main__":
    loader = ModelLoader()
    loader.load()
    print("Ready:", loader.is_ready())
