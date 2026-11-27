"""Train and save the German Credit classification pipeline."""

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "german.data"
MODEL_PATH = PROJECT_DIR / "model.joblib"
METRICS_PATH = PROJECT_DIR / "training_metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "German Credit Risk"
REGISTERED_MODEL_NAME = "GermanCreditRiskModel"
MODEL_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

COLUMN_NAMES = [
    "status_checking", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
    "present_residence", "property", "age", "other_installment_plans", "housing",
    "number_credits", "job", "people_liable", "telephone", "foreign_worker", "credit_risk",
]
TARGET_LABELS = {0: "Good Risk", 1: "Bad Risk"}


def main() -> None:
    """Fit a preprocessing and classification pipeline, then save its artifacts."""
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})

    X = df.drop(columns="credit_risk")
    y = df["credit_risk"]
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(**MODEL_PARAMS),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "random_state": RANDOM_STATE,
        "test_rows": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_bad_risk": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall_bad_risk": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_bad_risk": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model_input_example = X_train.head(3).copy()
    model_input_example[numeric_features] = model_input_example[numeric_features].astype("float64")
    with mlflow.start_run(run_name="logistic-regression-baseline"):
        mlflow.log_params(
            {
                **{f"classifier.{key}": value for key, value in MODEL_PARAMS.items()},
                "test_size": TEST_SIZE,
                "split_random_state": RANDOM_STATE,
                "stratify": True,
                "test_rows": metrics["test_rows"],
            }
        )
        mlflow.log_metrics(
            {
                key: float(metrics[key])
                for key in ("accuracy", "precision_bad_risk", "recall_bad_risk", "f1_bad_risk")
            }
        )
        mlflow.set_tags(
            {
                "dataset": "Statlog German Credit",
                "estimator": "LogisticRegression",
            }
        )
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="saved_model")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="german_credit_pipeline",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=model_input_example,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.compose._column_transformer._RemainderColsList",
            ],
        )

    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Target labels: 0 = Good Risk, 1 = Bad Risk")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Logged run to MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Registered model: {REGISTERED_MODEL_NAME}")


if __name__ == "__main__":
    main()
