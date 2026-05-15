import unittest
import sys
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from models.trainer import ModelTrainer
from features.text_processor import TextFeatureExtractor

class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = TextFeatureExtractor(max_features=100)
        cls.trainer = ModelTrainer()
        
        texts = [
            "Junior Python developer learning Docker",
            "Middle developer with 3 years experience in Kubernetes",
            "Senior architect with 10 years leading teams"
        ] * 10
        
        cls.X = cls.extractor.fit_transform(texts)
        cls.y = np.array([0, 1, 2] * 10)
    
    def test_logistic_regression_trains(self):
        metrics = self.trainer.train_logistic_regression(
            self.X, self.y, self.X, self.y
        )
        self.assertIn('f1_score', metrics)
        self.assertGreater(metrics['f1_score'], 0.5)
    
    def test_catboost_trains(self):
        metrics = self.trainer.train_catboost(
            self.X, self.y, self.X, self.y
        )
        self.assertIn('f1_score', metrics)
        self.assertGreater(metrics['f1_score'], 0.5)
    
    def test_model_saves(self):
        import tempfile
        import joblib
        
        self.trainer.train_logistic_regression(self.X, self.y, self.X, self.y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.trainer.save_models(tmpdir)
            self.assertTrue(os.path.exists(f"{tmpdir}/lr.joblib"))
            
            loaded_model = joblib.load(f"{tmpdir}/lr.joblib")
            pred = loaded_model.predict(self.X)
            self.assertEqual(len(pred), len(self.y))
    
    def test_prediction_shape(self):
        self.trainer.train_logistic_regression(self.X, self.y, self.X, self.y)
        model = self.trainer.models['lr']
        
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape, (30, 3))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

class TestModelPerformance(unittest.TestCase):
    def test_overfitting_detection(self):
        extractor = TextFeatureExtractor(max_features=50)
        trainer = ModelTrainer()
        
        train_texts = [f"Sample text number {i} for testing" for i in range(50)]
        test_texts = [f"Completely different text {i} for validation" for i in range(20)]
        
        X_train = extractor.fit_transform(train_texts)
        X_test = extractor.transform(test_texts)
        
        y_train = np.random.randint(0, 3, 50)
        y_test = np.random.randint(0, 3, 20)
        
        train_metrics = trainer.train_logistic_regression(X_train, y_train, X_train, y_train)
        test_metrics = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        f1_diff = train_metrics['f1_score'] - test_metrics['f1_score']
        self.assertLess(f1_diff, 0.5, f"Возможен оверфиттинг: разница F1 = {f1_diff:.2f}")

if __name__ == '__main__':
    unittest.main()
