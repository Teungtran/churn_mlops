from fastapi import UploadFile, HTTPException,Form,BackgroundTasks
from src.Churn.components.support import import_data,most_common,get_dummies
from typing_extensions import Optional
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from src.Churn.components.data_ingestion import DataIngestion
from src.Churn.config.configuration import ConfigurationManager, WebhookConfig
import joblib 
import mlflow
from src.Churn.utils.logging import logger
from src.Churn.utils.notify_webhook import post_to_webhook
from src.Churn.utils.visualize_ouput import visualize_customer_churn
from src.Churn.utils.Evidently import get_column_mapping,get_data_drift

from datetime import datetime
import time
import os
import dagshub
import tempfile
import os
web_hook_url = WebhookConfig().url

async def send_webhook_payload(
    message: str,
    avg_confidence: Optional[float] = None,
):
    try:
        logger.info("Preparing webhook notification")

        payload = {
            "message": message,
            "avg_confidence": avg_confidence,
        }

        await post_to_webhook(web_hook_url, payload)

    except Exception as e:
        logger.error(f"Webhook notification failed with unexpected error: {str(e)}")
class PredictionPipeline:
    def __init__(self, model_uri: str, scaler_uri: str,):
        pass
    
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=scaler_uri)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
                raise RuntimeError(f"Failed to load model or scaler: {e}")
    def process_data_for_churn(self,df_input: pd.DataFrame):
        df_input.columns = df_input.columns.map(str.strip)
        cols_to_drop = {"Age"}
        df_input.drop(columns=[col for col in cols_to_drop if col in df_input.columns], inplace=True)    
        df_input.dropna(inplace=True)
        if 'Price' not in df_input.columns:
            df_input['Price'] = df_input['Product Price']
        else:
            print("Price column already exists, skipping.") 
        df_input['TotalSpent'] = df_input['Quantity'] * df_input['Price']
        df_features = df_input.groupby("customer_id", as_index=False, sort=False).agg(
            LastPurchaseDate = ("Purchase Date","max"),
            Favoured_Product_Categories = ("Product Category", lambda x: most_common(list(x))),
            Frequency = ("Purchase Date", "count"),
            TotalSpent = ("TotalSpent", "sum"),
            Favoured_Payment_Methods = ("Payment Method", lambda x: most_common(list(x))),
            Customer_Name = ("Customer Name", "first"),
            Customer_Label = ("Customer_Labels", "first"),
        )
        df_features = df_features.drop_duplicates(subset=['Customer_Name'], keep='first')
        df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
        df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
        df_features['LastPurchaseDate'] = pd.to_datetime(df_features['LastPurchaseDate'])
        max_LastBuyingDate = df_features["LastPurchaseDate"].max()
        df_features['Recency'] = (max_LastBuyingDate - df_features['LastPurchaseDate']).dt.days
        df_features['LastPurchaseDate'] = df_features['LastPurchaseDate'].dt.date
        df_features['Avg_Spend_Per_Purchase'] = df_features['TotalSpent']/df_features['Frequency'].replace(0,1)
        df_features['Purchase_Consistency'] = df_features['Recency'] / df_features['Frequency'].replace(0, 1)
        df_features.drop(columns=["LastPurchaseDate"],axis=1,inplace=True)
        return df_features
    def encode_churn(self, df_features):
        df_copy = df_features.copy()
        df_copy.drop(columns=["customer_id","Customer_Name"],axis=1,inplace=True)
        df_features_encode = get_dummies(df_copy)
        return df_features_encode
    async def predict(self, reference_data: Optional[str] = None):
        try:
            start_time = time.time()
            start_datetime = datetime.now()
            time_str = start_datetime.strftime('%Y%m%dT%H%M%S')
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            mlflow_config = config_manager.get_mlflow_config()
            threshold_config = config_manager.get_threshold_config()
            dagshub.init(
            repo_owner="Teungtran",
            repo_name="churn_mlops",
            mlflow=True
        )
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            mlflow.set_experiment(mlflow_config.prediction_experiment_name)  
            with mlflow.start_run(run_name=f"prediction_run_{time_str}"):
                data_ingestion = DataIngestion(config=data_ingestion_config)
                df = data_ingestion.load_data()
                df_features = self.process_data_for_churn(df)
                df_encoded = self.encode_churn(df_features)
                drift_ratio = 0
                n_drifted_features = 0

                if reference_data:
                    try:
                        reference_df = pd.read_csv(reference_data)
                        column_mapings = get_column_mapping(df_features)
                        drift_result = get_data_drift(reference_df, df_features, column_mapings)
                        drift_ratio = drift_result.get("drift_ratio", 0)
                        n_drifted_features = drift_result.get("n_drifted_features", 0)
                    except Exception as e:
                        logger.warning(f"Skipping data drift due to error: {e}")
                else:
                    logger.info("No reference data provided. Skipping data drift detection.")      

                X = self.scaler.transform(df_encoded)
                y_pred = self.model.predict(X)
                df_features['Churn_RATE'] = y_pred
                counts = df_features['Churn_RATE'].value_counts()
                count_churn = counts.get(1, 0)
                count_not_churn = counts.get(0, 0)
                try:
                    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                        prediction_csv_path = temp_file.name
                        df_features.to_csv(prediction_csv_path, index=False)
                        mlflow.log_artifact(prediction_csv_path, "predictions")
                        logger.info(f"Successfully saved prediction results to {prediction_csv_path} and logged as MLflow artifact")
                    
                    os.remove(prediction_csv_path)
                    logger.info(f"Deleted temporary prediction file: {prediction_csv_path}")

                except Exception as e:
                    logger.error(f"An error occurred during prediction saving or cleanup: {e}")
                
                try:    
                    sklearn_model = self.model._model_impl  
                    y_proba = sklearn_model.predict_proba(X)
                    max_confidence = y_proba.max(axis=1)
                    average_confidence = max_confidence.mean()
                except AttributeError:
                    average_confidence = None 
                end_time = time.time()
                end_datetime = datetime.now()
                processing_time = end_time - start_time

                logger.info(f"Prediction processing time: {processing_time:.2f} seconds")
                logger.info(f"Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Completed at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                plot_path = visualize_customer_churn(df_features)
                mlflow.log_artifact(plot_path, "visualization")
                os.remove(plot_path)
                mlflow.log_metric("processing_time_seconds", processing_time)
                mlflow.log_metric("count_churn", count_churn)
                mlflow.log_metric("count_not_churn", count_not_churn)
                mlflow.log_param("start_time", start_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("end_time", end_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("rawdata_records", len(df))
                mlflow.log_metric("records_processed", len(df_encoded))
                message = ""
                if reference_data:
                    DRIFTED_FEATURE_THRESHOLD = threshold_config.data_drift_threshold
                    if drift_ratio > DRIFTED_FEATURE_THRESHOLD:
                        message += (
                            f"⚠️ Data drift detected in {n_drifted_features} feature(s) "
                            f"({drift_ratio:.0%} of all features). Consider retraining the model.\n"
                        )
                    else:
                        message += f"No significant data drift detected ({drift_ratio:.0%} of features).\n"
                else:
                    message = ""

                CONFIDENCE_THRESHOLD = threshold_config.confidence_threshold
                if average_confidence is not None:
                    mlflow.log_metric("average_prediction_confidence", average_confidence)

                    if average_confidence < CONFIDENCE_THRESHOLD:
                        message += (
                            f"⚠️ Average prediction confidence ({average_confidence:.2%}) is below the threshold "
                            f"of {CONFIDENCE_THRESHOLD:.2%}. Consider retraining the model."
                        )
                    else:
                        message += (
                            f"✅ Average prediction confidence ({average_confidence:.2%}) is above the threshold "
                            f"of {CONFIDENCE_THRESHOLD:.2%}. No further action required."
                        )

                    await send_webhook_payload(message=message, avg_confidence=average_confidence)

                mlflow.log_text(message, "prediction_summary.txt")
            return message

        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")
        
async def run_prediction_task(
    file_path: str,
    model_version: str,
    scaler_version: str,
    run_id: str,
    reference_data: Optional[str] = None
):
    """
    Background task to run prediction pipeline
    """
    try:
        model_uri = f"models:/RandomForestClassifier/{model_version}"
        scaler_uri = f"runs:/{run_id}/{scaler_version}"
        pipeline = PredictionPipeline(model_uri, scaler_uri)
        message = await pipeline.predict(reference_data=reference_data) 

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleanup: Deleted input file {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete input file during cleanup: {e}")

        return message

    except Exception as e:
        logger.error(f"Background prediction task error: {e}")
        return f"Prediction error: {e}"


class ChurnController:
    @staticmethod
    async def predict_churn(
        background_tasks: BackgroundTasks,
        file: UploadFile,
        model_version: str = Form(default="1"),
        scaler_version: str = Form(default="scaler_churn_version_20250701T105905.pkl"),
        run_id: str = Form(default="b523ba441ea0465085716dcebb916294"),
        reference_data: Optional[str] = Form(default=None),
    ):
        """
        Predict churn using uploaded file and dynamic model/scaler versions.
        """
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        input_file_path = data_ingestion_config.local_data_file

        try:
            await import_data(file)
            background_tasks.add_task(
                run_prediction_task,
                file_path=input_file_path,
                model_version=model_version,
                scaler_version=scaler_version,
                run_id=run_id,
                reference_data=reference_data
            )
            
            message = "Prediction task started in background. Results will be saved to experiment 'Churn_model_prediction_cycle' in https://dagshub.com/Teungtran/churn_mlops.mlflow, check your webhook for status "
            
            return {
                "message": message
            }

        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")