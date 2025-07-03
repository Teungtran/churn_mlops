import os
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import joblib as jb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,precision_recall_curve, average_precision_score,matthews_corrcoef
)
from pathlib import Path
import mlflow
import mlflow.sklearn
from src.Churn.utils.logging import logger
from src.Churn.entity.config_entity import TrainingConfig, EvaluationConfig
from src.Churn.utils.common import save_json

class TrainAndEvaluateModel:
    def __init__(self, config_train: TrainingConfig, config_eval: EvaluationConfig = None):
        self.train_config = config_train
        self.eval_config = config_eval
        self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')
        self.model_name = f"model_churn_{self.datetime_suffix}"
        self.fine_tuned_model_name = f"finetuned_churn_{self.datetime_suffix}"
    def log_model_to_mlflow(self, model, model_name: str):
        logger.info(f"Logging model to MLflow with artifact path: {model_name}")
        try:
            mlflow.sklearn.log_model(model, artifact_path=model_name)

            run_id = mlflow.active_run().info.run_id

            artifact_uri = f"runs:/{run_id}/{model_name}"

            registered_model_name = "RandomForestClassifier"
            mlflow.register_model(model_uri=artifact_uri, name=registered_model_name)

            logger.info(f"✅ Successfully registered model as '{registered_model_name}' from: {artifact_uri}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to log or register model to MLflow: {e}")
            logger.warning("Continuing without MLflow model registration")
        
    def train(self, X_train_scaled, y_train, model):
        """Train the model."""
        logger.info(f"Loading model: {model}")
        logger.info(f"Starting model training for {self.model_name}")
        trained_model = model.fit(X_train_scaled, y_train)

        logger.info(f"Model training for {self.model_name} completed")
        model_version_dir = str(self.train_config.model_version_dir)
        os.makedirs(model_version_dir, exist_ok=True)
        
        trained_model_path_versioned = os.path.join(model_version_dir, f"model_churn_version_{self.datetime_suffix}.pkl")
        
        jb.dump(trained_model, trained_model_path_versioned)
        logger.info(f"  trained model file (for future use): {trained_model_path_versioned}")
        return trained_model, trained_model_path_versioned
    
    def fine_tune(self, trained_model, X_train_scaled, y_train):
        """Fine-tune the exact trained model with hyperparameter search."""
        logger.info("Starting fine-tuning of the trained model")
        
        rf_params = {
            'n_estimators': [100, 200, 300, 400, 500, 700, 1000],  
            'criterion': ['gini', 'entropy', 'log_loss'],          # log_loss for classification since sklearn 1.1+
            'max_depth': [None, 10, 20, 30, 50, 70],                # Include deeper trees
            'min_samples_split': [2, 5, 10, 15],                    # More control over overfitting
            'min_samples_leaf': [1, 2, 4, 6],                       # Helps with generalization
            'max_features': ['sqrt', 'log2', None],                # 'auto' is deprecated; None = all features
            'bootstrap': [True, False],                             # Evaluate both bootstrapped and full datasets
            'class_weight': [None, 'balanced', 'balanced_subsample']  # Handles imbalanced datasets
        }

        logger.info(f"Fine-tuning model with current parameters: n_estimators={trained_model.n_estimators}, criterion={trained_model.criterion}")
        random_search = RandomizedSearchCV(trained_model, rf_params, cv=5, n_jobs=1, n_iter=20, random_state=42)
        random_search.fit(X_train_scaled, y_train)
        
        best_model = RandomForestClassifier(**random_search.best_params_, random_state=42)
        logger.info(f"Best parameters found: {random_search.best_params_}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
        # Fit the best model with the training data
        best_model = best_model.fit(X_train_scaled, y_train)
        logger.info("Best model fitted with training data")

        model_version_dir = str(self.train_config.model_version_dir)
        os.makedirs(model_version_dir, exist_ok=True)
        
        fine_tuned_model_path_versioned = os.path.join(model_version_dir, f"finetuned_churn_{self.datetime_suffix}.pkl")
        
        jb.dump(best_model, fine_tuned_model_path_versioned)
        
        logger.info(f"Fine-tuned models saved:")
        logger.info(f"  Versioned file (for future use): {fine_tuned_model_path_versioned}")
        
        return best_model, fine_tuned_model_path_versioned
    
    def perform_detailed_evaluation(self, model, X_test_scaled, y_test):
        """Evaluate the model in detail and save metrics."""
        logger.info("Performing detailed evaluation on test data")

        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        avg_precision = average_precision_score(y_test, y_pred_prob)
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "mcc": float(mcc),
            "avg_precision": float(avg_precision)
        }
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report
        os.makedirs(self.eval_config.evaluation_dir, exist_ok=True)
        metrics_file_versioned = Path(self.eval_config.evaluation_dir) / f"metrics_{self.datetime_suffix}.json"
        
        save_json(metrics_file_versioned, metrics)
        logger.info(f"Detailed metrics saved to: {metrics_file_versioned}")
        
        # Log all metrics to MLflow
        mlflow.log_metrics({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "mcc": mcc,
            "avg_precision": avg_precision
        })
        
        # Log additional metrics from classification report
        for class_label, class_metrics in report.items():
            if isinstance(class_metrics, dict):  # Skip 'accuracy', etc.
                for metric_name, value in class_metrics.items():
                    if isinstance(value, (int, float)):  # Only log numeric values
                        mlflow.log_metric(f"class_{class_label}_{metric_name}", value)
        
        mlflow.log_artifact(str(metrics_file_versioned))
        return metrics, y_pred, y_pred_prob
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot and save confusion matrix."""
        logger.info("Creating confusion matrix plot")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(self.eval_config.evaluation_dir, exist_ok=True)
        cm_path = os.path.join(self.eval_config.evaluation_dir, f"confusion_matrix_{self.datetime_suffix}.png")
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Confusion matrix saved to: {cm_path}")
        mlflow.log_artifact(cm_path)

        return cm_path
    def plot_precision_recall_curve(self, y_test, y_pred_prob):
        """Plot and save Precision-Recall curve."""
        logger.info("Creating Precision-Recall curve plot")

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
        avg_precision = average_precision_score(y_test, y_pred_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='purple', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')

        # Save plot with datetime suffix naming
        os.makedirs(self.eval_config.evaluation_dir, exist_ok=True)
        pr_path = os.path.join(self.eval_config.evaluation_dir, f"precision_recall_curve_{self.datetime_suffix}.png")
        plt.savefig(pr_path)
        plt.close()
        logger.info(f"Precision-Recall curve saved to: {pr_path}")
        mlflow.log_artifact(pr_path)
        return pr_path
    def plot_roc_curve(self, y_test, y_pred_prob):
        """Plot and save ROC curve."""
        logger.info("Creating ROC curve plot")
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the plot with datetime naming
        os.makedirs(self.eval_config.evaluation_dir, exist_ok=True)
        roc_path = os.path.join(self.eval_config.evaluation_dir, f"roc_curve_{self.datetime_suffix}.png")
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"ROC curve saved to: {roc_path}")
        mlflow.log_artifact(roc_path)

        return roc_path
    
    def train_and_evaluate(self, base_model, X_train_scaled, X_test_scaled, y_train, y_test):
        """Main method to train and evaluate the model using temporary directories."""
        logger.info("Initiating model training and evaluation with temporary directory management")

        
        try:
            try:
                logger.info(f"Loaded y_train shape: {y_train.shape}")
                logger.info(f"Loaded y_test shape: {y_test.shape}")
                
                # Handle NaN values in target variables
                logger.info(f"NaN count in y_train: {y_train.isna().sum()}")
                logger.info(f"NaN count in y_test: {y_test.isna().sum()}")
                
                if y_train.isna().sum() > 0:
                    logger.warning(f"Found {y_train.isna().sum()} NaN values in y_train, filling with 0")
                    y_train = y_train.fillna(0)
                    
                if y_test.isna().sum() > 0:
                    logger.warning(f"Found {y_test.isna().sum()} NaN values in y_test, filling with 0")
                    y_test = y_test.fillna(0)
                    
            except FileNotFoundError as e:
                logger.error(f"Target files not found: {e}")
                raise e

            logger.info(f"Final X_train_scaled shape: {X_train_scaled.shape}")
            logger.info(f"Final y_train shape: {y_train.shape}")
            logger.info(f"Final X_test_scaled shape: {X_test_scaled.shape}")
            logger.info(f"Final y_test shape: {y_test.shape}")

            model, trained_model_path = self.train(X_train_scaled, y_train, base_model)

            accuracy = model.score(X_test_scaled, y_test)
            mlflow.log_metric("accuracy after testing", accuracy)
            logger.info(f"Model accuracy on test data: {accuracy}")
            
            if accuracy < 0.85:
                logger.info("Model accuracy is less than 85%, fine-tuning needed")
                fine_tuned_model, fine_tuned_model_path = self.fine_tune(model, X_train_scaled, y_train)

                metrics, y_pred, y_pred_prob = self.perform_detailed_evaluation(fine_tuned_model, X_test_scaled, y_test)

                self.plot_confusion_matrix(y_test, y_pred)
                self.plot_precision_recall_curve(y_test, y_pred_prob)
                self.plot_roc_curve(y_test, y_pred_prob)
                accuracy = fine_tuned_model.score(X_test_scaled, y_test)
                logger.info(f"Model accuracy on test data after fine-tuned: {accuracy}")
                mlflow.log_metric("accuracy after fine-tuning", accuracy)
                self.log_model_to_mlflow(fine_tuned_model,str(self.fine_tuned_model_name))
                mlflow.log_artifact(str(trained_model_path),"Model before tunning")    
                return fine_tuned_model, metrics, fine_tuned_model_path
            else:
                logger.info("Model accuracy is 85% or above, not needed finetuned, log trained model to mlflow")
                self.log_model_to_mlflow(model,str(self.model_name))
                metrics, y_pred, y_pred_prob = self.perform_detailed_evaluation(model, X_test_scaled, y_test)
                
                self.plot_confusion_matrix(y_test, y_pred)
                self.plot_precision_recall_curve(y_test, y_pred_prob)
                self.plot_roc_curve(y_test, y_pred_prob)                
                return model, metrics, trained_model_path
        except Exception as e:
            logger.error(f"Error in model training and evaluation: {e}")
            raise e
                    