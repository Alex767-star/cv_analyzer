import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os

class ModelExplainer:
    def __init__(self, model, vectorizer, encoder):
        self.model = model
        self.vectorizer = vectorizer
        self.encoder = encoder
        
    def explain_prediction(self, text: str, top_features: int = 10):
        features = self.vectorizer.transform([text])
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features)
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[features.toarray().argmax(axis=1)[0]]
        else:
            shap_vals = shap_values[0]
        
        top_indices = np.argsort(np.abs(shap_vals))[-top_features:]
        
        plt.barh(
            [feature_names[i] for i in top_indices],
            [shap_vals[i] for i in top_indices]
        )
        plt.xlabel('SHAP value')
        plt.title('Влияние признаков на предсказание')
        plt.tight_layout()
        
        return plt.gcf()
    
    def generate_report(self, text: str) -> str:
        features = self.vectorizer.transform([text])
        prediction = self.model.predict_proba(features)[0]
        
        report = "SHAP Analysis Report\n"
        report += "=" * 50 + "\n"
        report += f"Predicted level: {self.encoder.inverse_transform([prediction.argmax()])[0]}\n"
        report += f"Confidence: {prediction.max():.2%}\n\n"
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features)
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[prediction.argmax()]
        else:
            shap_vals = shap_values
        
        top_features = np.argsort(np.abs(shap_vals[0]))[-5:][::-1]
        
        report += "Top 5 most influential features:\n"
        for idx in top_features:
            report += f"  {feature_names[idx]}: {shap_vals[0][idx]:.4f}\n"
        
        return report
