import os
import time
import json
import mlflow
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from evidently.pipeline.column_mapping import ColumnMapping
from src.Churn.utils.logging import logger


def get_data_drift(reference_df, df, output_dir: str = "plots"):
    """
    Calculate data drift between reference and current dataframes.
    
    Args:
        reference_df: Reference dataframe (historical data)
        df: Current dataframe (new data)
        output_dir: Directory to save reports
    
    Returns:
        Dictionary with drift metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reference data columns: {reference_df.columns.tolist()}")
    logger.info(f"Current data columns: {df.columns.tolist()}")

    common_columns = [col for col in reference_df.columns if col in df.columns]

    if not common_columns:
        logger.error("No common columns found between reference and current data")
        return {
            "drift_score": 1.0,
            "drift_ratio": 1.0,
            "n_drifted_features": 0,
            "total_features": 0
        }

    reference_subset = reference_df[common_columns].copy()
    current_subset = df[common_columns].copy()

    logger.info(f"Using {len(common_columns)} common columns for drift detection")

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = list(reference_subset.select_dtypes(include=["number"]).columns)
    column_mapping.categorical_features = list(reference_subset.select_dtypes(include=["object", "category"]).columns)

    drift_report = Report(metrics=[DatasetDriftMetric()])
    drift_report.run(reference_data=reference_subset, current_data=current_subset, column_mapping=column_mapping)

    timestamp = int(time.time())
    html_path = os.path.join(output_dir, f"Data_report_churn_{timestamp}.html")
    json_path = os.path.join(output_dir, f"Data_report_churn_{timestamp}.json")

    try:
        drift_report.save_html(html_path)
        logger.info(f"Saved HTML report to {html_path}")
        mlflow.log_artifact(html_path, artifact_path="drift_report")
        os.remove(html_path)
    except Exception as e:
        logger.warning(f"Could not save HTML report: {e}")

    drift_result_dict = drift_report.as_dict()
    with open(json_path, "w") as f:
        json.dump(drift_result_dict, f, indent=4)
    mlflow.log_artifact(json_path, artifact_path="drift_report")
    os.remove(json_path)

    # Parse results
    drift_summary = drift_result_dict["metrics"][0]["result"]
    drift_score = drift_summary.get("dataset_drift", None)
    n_drifted_features = drift_summary.get("number_of_drifted_columns", 0)
    total_features = drift_summary.get("number_of_columns", 0)
    drift_ratio = n_drifted_features / total_features if total_features > 0 else 0

    # Log to MLflow
    mlflow.log_metric("drift_score", drift_score)
    mlflow.log_metric("drift_ratio", drift_ratio)
    mlflow.log_metric("drifted_features", n_drifted_features)
    mlflow.log_metric("total_features", total_features)

    return {
        "drift_score": drift_score,
        "drift_ratio": drift_ratio,
        "n_drifted_features": n_drifted_features,
        "total_features": total_features
    }
