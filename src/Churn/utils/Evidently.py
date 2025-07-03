from evidently import Report, ColumnMapping
from evidently import DataDriftPreset
import time
import mlflow
import os
import json
def get_column_mapping(df):
    column_mapping = ColumnMapping(
    target=None,
    prediction=None,
    numerical_features=df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
    categorical_features=df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
)
    return column_mapping

def get_data_drift(reference_df, df, column_mapping, output_dir: str = "plots"):
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=df, column_mapping=column_mapping)

    # Save HTML Report
    timestamp = int(time.time())
    html_report_path = os.path.join(output_dir, f"Data_report_churn_{timestamp}.html")
    drift_report.save_html(html_report_path)
    mlflow.log_artifact(html_report_path, artifact_path="drift_report")
    os.remove(html_report_path)

    drift_result_dict = drift_report.as_dict()
    json_report_path = os.path.join(output_dir, f"Data_report_churn_{timestamp}.json")
    with open(json_report_path, "w") as f:
        json.dump(drift_result_dict, f, indent=4)
    mlflow.log_artifact(json_report_path, artifact_path="drift_report")
    os.remove(json_report_path)

    drift_summary = drift_result_dict["metrics"][0]["result"]
    drift_score = drift_summary.get("dataset_drift", None)
    n_drifted_features = drift_summary.get("number_of_drifted_features", 0)
    total_features = drift_summary.get("number_of_features", 0)

    mlflow.log_metric("drift_score", float(drift_score))
    mlflow.log_metric("drifted_features", n_drifted_features)
    mlflow.log_metric("total_features", total_features)
    result = {
        "drift_score": drift_score,
        "n_drifted_features": n_drifted_features,
        "total_features": total_features
    }
    return result