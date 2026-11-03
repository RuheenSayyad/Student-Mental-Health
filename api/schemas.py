from typing import Literal, Optional
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    student_id: Optional[str] = "STU001"
    gender: Literal["Male", "Female"]
    age: int = Field(ge=15, le=65)
    course: str = Field(description="Degree or program of study, e.g. Engineering, BCS, Law")
    year_of_study: str = Field(description="e.g. year 1, year 2, year 3, year 4")
    cgpa_range: str = Field(description="e.g. 3.00 - 3.49, 3.50 - 4.00, 2.50 - 2.99")
    marital_status: Literal["Yes", "No"]
    anxiety: Literal["Yes", "No"]
    panic_attack: Literal["Yes", "No"]
    specialist_treatment: Literal["Yes", "No"]


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 for No Depression, 1 for Depression")
    depression_probability: float = Field(description="Model predicted probability of depression")
    risk_level: Literal["Low", "Medium", "High"]


class ErrorDetail(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    message: str
    details: list[ErrorDetail]


class BatchPredictionRequest(BaseModel):
    students: list[StudentInput] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
