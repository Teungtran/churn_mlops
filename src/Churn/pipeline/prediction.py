from fastapi import UploadFile, HTTPException,Form
from src.Churn.components.support import import_data,df_to_records,most_common,get_dummies
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from src.Churn.components.data_ingestion import DataIngestion
from src.Churn.config.configuration import ConfigurationManager
import joblib 
import mlflow
from src.Churn.utils.logging import logger
from datetime import datetime
import time
import os

version_model_path = f"models:/RandomForestClassifier/1"
scaler_uri_path = f"runs:/f3ab09385e414fd2abf29d80f74cd67a/scaler_churn_version_20250625T154746.pkl"

try:
    loaded_model = mlflow.pyfunc.load_model(version_model_path)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=scaler_uri_path)
    loaded_scaler = joblib.load(local_path)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    loaded_model = None
    scaler_churn = None



class PredictionPipeline:
    def __init__(self, model_uri: str, scaler_uri: str):
        mlflow.set_tracking_uri("https://dagshub.com/Teungtran/churn_mlops.mlflow")

        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=scaler_uri)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
                raise RuntimeError(f"Failed to load model or scaler: {e}")
    def process_data_for_churn(self,df_input: pd.DataFrame):
        df_input.columns = df_input.columns.map(str.strip)
        cols_to_drop = {"Returns", "Age", "Total Purchase Amount"}
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
        df_features.drop(columns=["customer_id","LastPurchaseDate",'Customer_Name'],axis=1,inplace=True)
        
        return df_features
    def encode_churn(self,df_features: pd.DataFrame):
        df_copy = df_features.copy()
        df_features_encode = get_dummies(df_copy)
        return df_features_encode
    def predict(self):
        try:
            start_time = time.time()
            start_datetime = datetime.now()
            time_str = start_datetime.strftime('%Y%m%dT%H%M%S')
            
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            
            data_ingestion = DataIngestion(config=data_ingestion_config)
            
            df = data_ingestion.load_data()
            df_processed = self.process_data_for_churn(df)
            df_encoded = self.encode_churn(df_processed)
            X = self.scaler.transform(df_encoded)

            y_pred = self.model.predict(X)
            df['Churn_RATE'] = y_pred

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

            with mlflow.start_run(run_name=f"prediction_run_{time_str}"):
                mlflow.log_metric("processing_time_seconds", processing_time)
                mlflow.log_param("start_time", start_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("end_time", end_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("rawdata_records", len(df))
                mlflow.log_metric("records_processed", len(df_processed))
                if average_confidence is not None:
                    mlflow.log_metric("average_prediction_confidence", average_confidence)
            
            # Delete the input raw data file after prediction
            if os.path.exists(data_ingestion_config.local_data_file):
                try:
                    os.remove(data_ingestion_config.local_data_file)
                    logger.info(f"Successfully deleted input raw data file: {data_ingestion_config.local_data_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete input raw data file: {e}")

            return df, average_confidence, {
                "processing_time_seconds": processing_time,
                "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                "records_processed": len(df),
                "average_confidence": average_confidence
            }

        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")


class ChurnController:
    @staticmethod
    async def predict_churn(
        file: UploadFile,
        model_version: str = Form(default="1"),
        scaler_version: str = Form(default="scaler_churn_version_20250625T154746.pkl"),
        run_id: str = Form(default="f3ab09385e414fd2abf29d80f74cd67a"),
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

            model_uri = f"models:/RandomForestClassifier/{model_version}"
            scaler_uri = f"runs:/{run_id}/{scaler_version}"

            pipeline = PredictionPipeline(model_uri, scaler_uri)
            df, avg_confidence, timing_info = pipeline.predict()

            response = {
                "results": df_to_records(df),
                "timing": timing_info
            }

            CONFIDENCE_THRESHOLD = 0.7 
            if avg_confidence is not None and avg_confidence < CONFIDENCE_THRESHOLD:
                warning_msg = (
                    f"⚠️ Average prediction confidence ({avg_confidence:.2%}) is below the threshold "
                    f"of {CONFIDENCE_THRESHOLD:.2%}. Consider retraining the model."
                )
                response["warning"] = warning_msg
                logger.warning(warning_msg)

            return response

        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
        finally:
            # Ensure input file is deleted even if an exception occurs
            if os.path.exists(input_file_path):
                try:
                    os.remove(input_file_path)
                    logger.info(f"Cleanup: Deleted input file {input_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete input file during cleanup: {e}")