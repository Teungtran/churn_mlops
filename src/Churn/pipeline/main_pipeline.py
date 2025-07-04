from src.Churn.config.configuration import ConfigurationManager
from .prepare_data import DataPreparationPipeline
from .prepare_model import ModelPreparationPipeline
from .train_evaluation import TrainEvaluationPipeline
from .cloud_storage_push import CloudStoragePushPipeline
from src.Churn.utils.logging import logger
from .cleanup import cleanup_temp_files
from src.Churn.components.support import import_data
from pathlib import Path
from fastapi import UploadFile
import mlflow
import dagshub
from datetime import datetime

class WorkflowRunner:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.uploaded_file = None

    def check_data_file_exists(self):
        """Check if the local data file exists and has content"""
        try:
            config = self.config_manager.get_data_ingestion_config()
            data_file_path = Path(config.local_data_file)
            
            if data_file_path.exists() and data_file_path.stat().st_size > 0:
                logger.info(f"Data file found: {data_file_path}")
                return True
            else:
                logger.info(f"Data file not found or empty: {data_file_path}")
                return False
        except Exception as e:
            logger.error(f"Error checking data file: {e}")
            return False
            
    async def run(self, uploaded_file: UploadFile = None):
        """Run the complete workflow with proper path passing between stages."""
        self.uploaded_file = uploaded_file
        
        try:
            mlflow_config = self.config_manager.get_mlflow_config()
            logger.info(f"MLflow configured with experiment: {mlflow_config.experiment_name}")
            dagshub.init(
                repo_owner=mlflow_config.dagshub_username,
                repo_name=mlflow_config.dagshub_repo_name,
                mlflow=True
            )
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            mlflow.set_experiment(mlflow_config.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow configuration failed: {e}. Continuing without MLflow tracking.")
            mlflow_config = None

        if not self.check_data_file_exists():
            if self.uploaded_file is not None:
                logger.info("=" * 50)
                logger.info("STAGE 0: Data Import")
                logger.info("=" * 50)
                
                try:
                    await import_data(self.uploaded_file)
                    logger.info("Data import completed successfully")
                except Exception as e:
                    logger.error(f"Data import failed: {e}")
                    raise
            else:
                raise ValueError("No data file found and no uploaded file provided. Please provide a data file.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"Churn_model_training_cycle_{timestamp}"
        with mlflow.start_run(run_name=run_name):
            
            logger.info("=" * 50)
            logger.info("STAGE 1: Data Preparation")
            logger.info("=" * 50)
            
            data_prep = DataPreparationPipeline()
            X_train, X_test, y_train, y_test, df_processed, df, df_features = data_prep.main()
            
            logger.info("Data preparation completed successfully")

            # Log raw dataframe
            raw_dataset = mlflow.data.from_pandas(
                df=df,
                name="churn-prediction-raw-training-data",
                targets="Churn"
            )
            mlflow.log_input(raw_dataset, context="train_data_raw")

            feature_dataset = mlflow.data.from_pandas(
                df=df_features,
                name="churn-prediction-feature-data",
                targets="Churn"
            )
            mlflow.log_input(feature_dataset, context="train_data_features")
            
            encode_dataset = mlflow.data.from_pandas(
                df=df_processed,
                name="churn-prediction-training-data-encoded",
                targets="Churn"
            )
            mlflow.log_input(encode_dataset, context="train_data_encoded")
            logger.info("=" * 50)
            logger.info("STAGE 2: Model Preparation")
            logger.info("=" * 50)
            
            mlflow_config = self.config_manager.get_mlflow_config()
            model_prep = ModelPreparationPipeline(mlflow_config=mlflow_config)
            model, base_model_path, scaler_path, X_train_scaled, X_test_scaled = model_prep.main(
                X_train=X_train,
                X_test=X_test
            )
            
            logger.info("Model preparation completed successfully")
            logger.info(f"Base model: {base_model_path}")
            logger.info(f"Scaler: {scaler_path}")

            logger.info("=" * 50)
            logger.info("STAGE 3: Train and Evaluate")
            logger.info("=" * 50)
            
            train_eval = TrainEvaluationPipeline(mlflow_config=mlflow_config)
            model, metrics, final_model_path = train_eval.main(
                base_model=model,
                X_train_scaled=X_train_scaled,
                X_test_scaled=X_test_scaled,
                y_train=y_train,
                y_test=y_test
            )
            
            logger.info("Training and evaluation completed successfully")
            logger.info(f"Metrics: {metrics}")
        
        # Stage 4: Cloud Storage Push (outside the MLflow run)
        logger.info("=" * 50)
        logger.info("STAGE 4: Cloud Storage Push")
        logger.info("=" * 50)
        
        cloud_push = CloudStoragePushPipeline()
        cloud_push.main()
        logger.info("Cloud storage push completed successfully")

        cleanup_temp_files()
        logger.info("=" * 50)
        logger.info("STAGE 5: Cleanup file")
        logger.info("=" * 50)
        logger.info("=" * 50)
        logger.info(f"WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)

        return final_model_path