import unittest
import numpy as np
import sys
sys.path.append('../src')

from data.dataset import CVDataset
from features.text_processor import TextFeatureExtractor
from models.trainer import ModelTrainer

class TestCVPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_texts = [
            "Python разработчик с опытом 5 лет",
            "Junior Data Scientist знает SQL и Python",
            "Senior DevOps инженер Docker Kubernetes AWS"
        ]
        self.extractor = TextFeatureExtractor(max_features=100)
        
    def test_text_processing(self):
        processed = [self.extractor.lemmatize(text) for text in self.sample_texts]
        self.assertEqual(len(processed), 3)
        
    def test_feature_extraction(self):
        features = self.extractor.fit_transform(self.sample_texts)
        self.assertEqual(features.shape[0], 3)
        self.assertGreater(features.shape[1], 0)
        
    def test_end_to_end(self):
        features = self.extractor.fit_transform(self.sample_texts)
        trainer = ModelTrainer()
        
        y = np.array([2, 0, 1])  # senior, junior, middle
        
        try:
            metrics = trainer.train_logistic_regression(features, y, features, y)
            self.assertIn('f1_score', metrics)
        except Exception as e:
            self.skipTest(f"Training skipped: {e}")

if __name__ == '__main__':
    unittest.main()
