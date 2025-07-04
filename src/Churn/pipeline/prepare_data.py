from src.Churn.config.configuration import ConfigurationManager
from src.Churn.components.data_ingestion import DataIngestion
from src.Churn.utils.logging import logger


STAGE_NAME = "Data Ingestion stage"
class DataPreparationPipeline:
    def main(self):
        logger.info(f">>> Stage {STAGE_NAME} started <<<")
        
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        X_train, X_test, y_train, y_test,df_processed,df,df_features = data_ingestion.data_ingestion_pipeline()
        
        logger.info(f">>> Stage {STAGE_NAME} completed <<<")
        return X_train, X_test, y_train, y_test,df_processed,df ,df_features

