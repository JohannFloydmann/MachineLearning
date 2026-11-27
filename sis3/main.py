"""FastAPI application for German Credit risk predictions."""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError("model.joblib is missing. Run `python train.py` before starting the API.")

model = joblib.load(MODEL_PATH)
app = FastAPI(
    title="German Credit Risk API",
    description="A Logistic Regression pipeline trained on the Statlog German Credit dataset.",
    version="1.0.0",
)


class CreditApplication(BaseModel):
    """The 20 application features expected by the trained model."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "status_checking": "A11", "duration_months": 6, "credit_history": "A34",
        "purpose": "A43", "credit_amount": 1169, "savings": "A65",
        "employment_since": "A75", "installment_rate": 4, "personal_status_sex": "A93",
        "other_debtors": "A101", "present_residence": 4, "property": "A121", "age": 67,
        "other_installment_plans": "A143", "housing": "A152", "number_credits": 2,
        "job": "A173", "people_liable": 1, "telephone": "A192", "foreign_worker": "A201",
    }})

    status_checking: str = Field(description="Checking account status (A11-A14)")
    duration_months: int = Field(ge=1, description="Credit duration in months")
    credit_history: str
    purpose: str
    credit_amount: int = Field(gt=0)
    savings: str
    employment_since: str
    installment_rate: int = Field(ge=1, le=4)
    personal_status_sex: str
    other_debtors: str
    present_residence: int = Field(ge=1, le=4)
    property: str
    age: int = Field(ge=18)
    other_installment_plans: str
    housing: str
    number_credits: int = Field(ge=1)
    job: str
    people_liable: int = Field(ge=1)
    telephone: str
    foreign_worker: str


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(description="0 = Good Risk; 1 = Bad Risk")
    risk_label: Literal["Good Risk", "Bad Risk"]


@app.get("/")
def read_root() -> dict[str, str]:
    """Confirm that the prediction API is running."""
    return {"message": "ML API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    """Return the predicted credit risk class for one application."""
    features = pd.DataFrame([application.model_dump()])
    predicted_class = int(model.predict(features)[0])
    label = "Bad Risk" if predicted_class == 1 else "Good Risk"
    return PredictionResponse(prediction=predicted_class, risk_label=label)
