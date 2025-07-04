from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
from src.Churn.pipeline.prediction import ChurnController
from typing_extensions import Optional
router = APIRouter(
    prefix="/churn",
    tags=["Churn Prediction"],
    responses={404: {"description": "Not found"}},
)


class ChurnResponse(BaseModel):
    message: str

@router.post("/", response_model=ChurnResponse)
async def predict_churn(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_version: str = Form(default="1"),
    scaler_version: str = Form(default="scaler_churn_version_20250701T105905.pkl"),
    run_id: str = Form(default="b523ba441ea0465085716dcebb916294")):
    return await ChurnController.predict_churn(background_tasks=background_tasks, file=file, model_version=model_version, scaler_version=scaler_version, run_id=run_id)
