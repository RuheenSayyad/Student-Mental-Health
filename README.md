# Student Mental Health - Depression Risk Prediction

Machine-learning project that predicts whether a university student reports depression from survey data
(demographics, academics, anxiety, panic attacks, specialist treatment). It follows the same lab-by-lab
structure as the `churn-prediction` project: baseline -> experiment tracking -> validated preprocessing
pipeline -> model registry.

> This is a small survey dataset (101 students; 26 in the test split), used for learning. Metrics move a lot
> with a single student, and the model is a screening exercise, not a diagnostic tool.

## Project structure

```text
Student_Mental_Health_ML_Project/
+-- configs/config.yaml            # dataset, split, MLflow and path settings (read by src/utils.py)
+-- data/
|   +-- raw/Student_Mental_Health.csv
|   +-- processed/                 # X_*_final.npy, y_*.npy, dataset_metadata.json
+-- src/
|   +-- utils.py                   # shared config, cleaning, feature lists, stratified split
|   +-- preprocess.py              # Lab 3/4  baseline preprocessing (scaler + one-hot)
|   +-- train.py                   # Lab 3    Random Forest baseline
|   +-- evaluate.py                # Lab 3    metrics + false_negatives / false_positives CSVs
|   +-- train_mlflow.py            # Lab 4    MLflow runs (RF, RF shallow, Logistic Regression)
|   +-- validate_reproducibility.py# Lab 4    two identical trainings must give identical F1
|   +-- validate_data.py           # Lab 5    pandera schema check of the raw CSV
|   +-- preprocess_pipeline.py     # Lab 5    sklearn ColumnTransformer -> models/preprocessor.pkl
|   +-- validate_outputs.py        # Lab 5    checks NaNs, shapes, binary target, class balance
|   +-- train_registry.py          # Lab 6    trains + registers a model version
|   +-- automate_lifecycle.py      # Lab 6    champion / challenger promotion by recall
|   +-- generate_registry_report.py# Lab 6    artifacts/production_model_report.json
+-- pipelines/run_lab3_baseline.py, run_lab4_tracking.py, run_lab5_pipeline.py, run_lab6_registry.py
+-- notebooks/
|   +-- project_implementation.ipynb        # end-to-end walkthrough (like the churn notebook)
+-- tests/test_pipeline.py
+-- models/  outputs/  artifacts/  reports/  logs/   # generated files
+-- main.py  requirements.txt  .gitignore
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Running

Always run from the project root.

| Lab | Command | What it does |
| :-- | :-- | :-- |
| 3 - Baseline | `python pipelines/run_lab3_baseline.py` | preprocess -> train RF -> evaluate + error analysis |
| 4 - Tracking | `python pipelines/run_lab4_tracking.py` | preprocess -> 3 MLflow runs -> reproducibility check |
| 5 - Pipeline | `python pipelines/run_lab5_pipeline.py` | validate raw data -> sklearn pipeline -> validate outputs |
| 6 - Registry | `python pipelines/run_lab6_registry.py` | register model -> promote champion -> deployment report |

`python main.py` runs Labs 3 -> 6 in order; `python main.py --lab 5` runs one lab.
Lab 6 needs the Lab 5 outputs (`models/preprocessor.pkl` and the processed arrays).
To browse experiments and the registry: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Tests: `python -m unittest discover -s tests -p "test_*.py"`

## Data preparation

- Column headers shortened; `Timestamp` dropped.
- Year of study normalised (`year 1` / `Year 1` -> 1); CGPA bands (with stray spaces) mapped to midpoints.
- Engineered: `symptom_burden_score` (anxiety + panic attack), `senior_high_achiever`, and `faculty_group`
  (49 free-text course names grouped into 8 faculties).
- Target: `depression` (Yes -> 1, No -> 0), 35 of 101 students. Stratified 75 / 26 split, `random_state=42`.
- Lab 3/4 fill the one missing age with the overall median (like the churn `TotalCharges` fill);
  Lab 5 imputes inside the pipeline using training-set statistics only.

## Baseline result (Lab 3, Random Forest, test split)

| Accuracy | Precision | Recall | F1 | 
| :-: | :-: | :-: | :-: |
| 0.8077 | 0.8333 | 0.5556 | 0.6667 |

4 of 9 depressed students in the test split are missed (false negatives), which is why recall is the
decision metric for model promotion in Lab 6.
