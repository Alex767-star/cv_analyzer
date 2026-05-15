import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

class ModelEvaluator:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        axes[0].set_title('Confusion Matrix')
        
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df = metrics_df[metrics_df.index.isin(self.label_encoder.classes_)]
        metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=axes[1])
        axes[1].set_title('Per-class Metrics')
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png', dpi=150)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': np.mean(y_pred == y_test)
        }
