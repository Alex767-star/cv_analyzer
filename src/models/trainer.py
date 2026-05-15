import numpy as np
from typing import Dict, Any
import mlflow
import joblib
import logging
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, experiment_name: str = "cv_classifier"):
        self.models = {}
        self.metrics = {}
        
    def train_logistic_regression(self, X_train, y_train, X_val, y_val) -> Dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, roc_auc_score
        
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        metrics = {
            'f1_score': f1_score(y_val, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_val, y_proba, multi_class='ovr')
        }
        
        try:
            with mlflow.start_run(run_name="logistic_regression"):
                mlflow.log_metrics(metrics)
                mlflow.log_param("model_type", "logistic_regression")
        except:
            pass
        
        self.models['lr'] = model
        return metrics
    
    def train_catboost(self, X_train, y_train, X_val, y_val) -> Dict:
        from catboost import CatBoostClassifier
        from sklearn.metrics import f1_score, roc_auc_score
        
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            verbose=False
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        metrics = {
            'f1_score': f1_score(y_val, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_val, y_proba, multi_class='ovr')
        }
        
        try:
            with mlflow.start_run(run_name="catboost"):
                mlflow.log_metrics(metrics)
                mlflow.log_param("model_type", "catboost")
        except:
            pass
        
        self.models['catboost'] = model
        return metrics
    
    def save_models(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            save_path = f"{path}/{name}.joblib"
            joblib.dump(model, save_path)
            logger.info(f"Сохранена модель: {save_path}")
