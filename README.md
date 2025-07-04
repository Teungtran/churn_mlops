# Churn Prediction MLOps System

A complete MLOps system for customer churn prediction with real-time monitoring and model retraining capabilities.

## Features

- **Churn Prediction**: Upload customer data and get churn predictions with confidence scores
- **Real-time Monitoring**: WebSocket-based real-time updates for prediction results
- **Model Retraining**: Easily retrain models with new data
- **Data Drift Detection**: Automatically detect when data distributions change
- **Cloud Storage Integration**: Results stored in S3 for easy access
- **MLflow Integration**: Track experiments, models, and metrics

## System Architecture

The system consists of:

1. **FastAPI Backend**: Handles prediction requests, model retraining, and webhooks
2. **Streamlit Frontend**: User-friendly interface for interacting with the system
3. **WebSocket Connection**: Real-time updates between backend and frontend
4. **MLflow Tracking**: Experiment tracking and model registry
5. **S3 Storage**: Store prediction results and model artifacts

## Getting Started

### Prerequisites

- Docker
- AWS credentials (for S3 storage)

### Running with Docker

1. Build the Docker image:
   ```
   docker build -t churn-mlops .
   ```

2. Run the container:
   ```
   docker run -p 8888:8888 -p 8501:8501 churn-mlops
   ```

3. Access the Streamlit UI at `http://localhost:8501`
4. Access the FastAPI documentation at `http://localhost:8888/docs`

### Environment Variables

Create a `.env` file with the following variables:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
MLFLOW_TRACKING_URI=your_mlflow_tracking_uri
DAGSHUB_USERNAME=your_dagshub_username
DAGSHUB_TOKEN=your_dagshub_token
DAGSHUB_REPO_NAME=your_dagshub_repo_name
```

## Usage

### Predict Customer Churn

1. Navigate to the "Predict Churn" tab
2. Upload a CSV file with customer data
3. Configure model version and other parameters if needed
4. Click "Run Prediction"
5. View results in the "Webhook Monitor" tab when ready

### Retrain Model

1. Navigate to the "Retrain Model" tab
2. Check workflow status to ensure the system is ready
3. Upload new training data (optional)
4. Click "Start Retraining"
5. Wait for the process to complete

### Monitor Results

1. Navigate to the "Webhook Monitor" tab
2. View real-time updates from prediction runs
3. Download prediction results as CSV
4. Check for data drift warnings and model confidence metrics

## API Endpoints

- `/churn/`: Submit prediction requests
- `/workflow/train`: Trigger model retraining
- `/workflow/status`: Check workflow status
- `/webhook/notify`: Receive webhook notifications
- `/webhooks`: Get all webhook data
- `/ws`: WebSocket endpoint for real-time updates

## License

This project is licensed under the terms of the license included in the repository.