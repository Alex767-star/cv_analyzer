import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from features.text_processor import TextFeatureExtractor

class TestDataQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(os.path.dirname(__file__), '../data/training_data.csv')
        cls.df = pd.read_csv(data_path)
        cls.extractor = TextFeatureExtractor(max_features=1000)
    
    def test_data_not_empty(self):
        self.assertGreater(len(self.df), 0, "Датасет пуст")
    
    def test_required_columns(self):
        required = ['text', 'level']
        for col in required:
            self.assertIn(col, self.df.columns, f"Отсутствует колонка {col}")
    
    def test_level_classes(self):
        valid_levels = {'junior', 'middle', 'senior'}
        actual_levels = set(self.df['level'].unique())
        self.assertTrue(valid_levels.issuperset(actual_levels), 
                       f"Некорректные уровни: {actual_levels - valid_levels}")
    
    def test_no_empty_texts(self):
        empty_count = self.df['text'].isna().sum() + (self.df['text'] == '').sum()
        self.assertEqual(empty_count, 0, f"Найдено {empty_count} пустых текстов")
    
    def test_min_text_length(self):
        short_texts = self.df['text'].str.len() < 20
        self.assertLess(short_texts.sum(), len(self.df) * 0.05, 
                       "Слишком много коротких текстов")
    
    def test_class_balance(self):
        class_counts = self.df['level'].value_counts()
        min_class_ratio = class_counts.min() / class_counts.max()
        self.assertGreater(min_class_ratio, 0.1, 
                          f"Сильный дисбаланс классов: {min_class_ratio:.2f}")
    
    def test_feature_extraction(self):
        sample_texts = self.df['text'].head(100).tolist()
        features = self.extractor.fit_transform(sample_texts)
        self.assertGreater(features.shape[1], 0, "Нет признаков")
        self.assertEqual(features.shape[0], 100, "Не все тексты обработаны")
    
    def test_skills_extraction(self):
        text = "Python developer with Docker and Kubernetes experience"
        skills = self.extractor.extract_skills_keywords(text)
        self.assertIn('python', skills)
        self.assertIn('docker', skills)
        self.assertIn('kubernetes', skills)

if __name__ == '__main__':
    unittest.main()
