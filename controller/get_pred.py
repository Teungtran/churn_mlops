from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from src.Churn.pipeline.pred_pipeline import ChurnController
from typing_extensions import Optional, List, Dict, Any

router = APIRouter(
    prefix="/churn",
    tags=["Churn Prediction"],
    responses={404: {"description": "Not found"}},
)


class ChurnResponse(BaseModel):
    payload: Dict[str, Any]





@router.post("/", response_model=ChurnResponse)
async def predict_churn(
    file: UploadFile = File(...),
    model_version: str = Form(default="1"),
    scaler_version: str = Form(default="scaler_churn_version_20250701T105905.pkl"),
    run_id: str = Form(default="b523ba441ea0465085716dcebb916294"),
    reference_data: Optional[str] = Form(default=None)
    ) :
    
    result = await ChurnController.predict_churn(
        file=file, 
        model_version=model_version, 
        scaler_version=scaler_version, 
        run_id=run_id,
        reference_data=reference_data
    )
    
    # Handle different response types
    if isinstance(result, tuple):
        error_message, _ = result
        return {"payload": {"error": error_message}}
    elif isinstance(result, dict):
        return {"payload": result}
    else:
        return {"payload": {"message": str(result)}}
