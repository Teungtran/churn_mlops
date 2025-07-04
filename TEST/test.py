from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
from dotenv import load_dotenv
from TEST.prediction import router as prediction_router
from controller.retraining import router as retraining_router
from TEST.webhook import router as webhook_router
from TEST.webhook_receiver import router as webhook_receiver_router
# Load environment variables
load_dotenv()


# Create FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for customer churn prediction and model retraining",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction_router)
app.include_router(retraining_router)
app.include_router(webhook_router)
app.include_router(webhook_receiver_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Churn Prediction API. Use /docs to view the API documentation."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8800)
