"""Unit tests for cleaning, splitting and the processed artifacts.

Run from the project root:
    python -m unittest discover -s tests -p "test_*.py"
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (  # noqa: E402
    RAW_PATH, TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    clean_dataframe, load_and_split, categorize_course,
)


class TestCleaning(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv(RAW_PATH)

    def test_raw_data_not_empty(self):
        self.assertFalse(self.raw.empty)
        self.assertEqual(self.raw.shape[1], 11)

    def test_clean_columns_and_target(self):
        df = clean_dataframe(self.raw)
        self.assertEqual(list(df.columns), NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
        self.assertTrue(set(df[TARGET].unique()).issubset({0, 1}))
        self.assertEqual(len(df), len(self.raw))

    def test_imputation_modes(self):
        self.assertEqual(clean_dataframe(self.raw, impute_missing=True).isna().sum().sum(), 0)
        # raw data has one missing age; the pipeline mode must leave it for the imputer
        self.assertEqual(clean_dataframe(self.raw, impute_missing=False)["age"].isna().sum(), 1)

    def test_year_and_cgpa_are_normalised(self):
        df = clean_dataframe(self.raw)
        self.assertTrue(set(df["year_of_study"].unique()).issubset({1, 2, 3, 4}))
        self.assertTrue(df["cgpa_numeric"].between(1.0, 3.75).all())

    def test_symptom_burden_score(self):
        df = clean_dataframe(self.raw)
        expected = (df["anxiety"] == "Yes").astype(int) + (df["panic_attack"] == "Yes").astype(int)
        self.assertTrue((df["symptom_burden_score"] == expected).all())

    def test_course_grouping(self):
        self.assertEqual(categorize_course("BCS"), "Computing_IT")
        self.assertEqual(categorize_course("engin"), "Engineering")
        self.assertEqual(categorize_course("Laws"), "Law")
        self.assertEqual(categorize_course("Pendidikan islam"), "Islamic_Studies")


class TestSplit(unittest.TestCase):
    def test_split_is_stratified_and_reproducible(self):
        a = load_and_split()
        b = load_and_split()
        X_train, X_test, y_train, y_test = a
        self.assertEqual(len(X_train) + len(X_test), 101)
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), delta=0.05)
        pd.testing.assert_frame_equal(a[0], b[0])


@unittest.skipUnless(os.path.exists("data/processed/X_train_final.npy"), "run a preprocessing lab first")
class TestProcessedArtifacts(unittest.TestCase):
    def test_processed_arrays(self):
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])


@unittest.skipUnless(os.path.exists("models/preprocessor.pkl"), "run Lab 5 first")
class TestPreprocessorInference(unittest.TestCase):
    def test_preprocessor_transforms_new_student(self):
        import joblib
        pre = joblib.load("models/preprocessor.pkl")
        new = pd.DataFrame([{
            "age": 20, "year_of_study": 2, "cgpa_numeric": 3.25,
            "symptom_burden_score": 2, "senior_high_achiever": 0,
            "gender": "Female", "marital_status": "No", "anxiety": "Yes",
            "panic_attack": "Yes", "specialist_treatment": "No",
            "faculty_group": "Computing_IT",
        }])
        out = pre.transform(new)
        self.assertEqual(out.shape[0], 1)
        self.assertFalse(np.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
