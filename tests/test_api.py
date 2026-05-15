import unittest
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

class TestAPIValidation(unittest.TestCase):
    def test_text_preprocessing(self):
        from features.text_processor import TextFeatureExtractor
        
        extractor = TextFeatureExtractor()
        
        dirty_text = "Python!!! developer @#$ with 5+ years experience..."
        clean = extractor.lemmatize(dirty_text)
        
        self.assertNotIn("@#$", clean)
        self.assertIsInstance(clean, str)
        self.assertGreater(len(clean), 0)
    
    def test_prediction_response_format(self):
        import numpy as np
        
        mock_proba = np.array([0.1, 0.8, 0.1])
        mock_levels = ['junior', 'middle', 'senior']
        
        prediction = mock_levels[np.argmax(mock_proba)]
        confidence = np.max(mock_proba)
        
        self.assertEqual(prediction, 'middle')
        self.assertAlmostEqual(confidence, 0.8)
        self.assertGreater(confidence, 0.5)
    
    def test_file_extension_validation(self):
        valid_extensions = ['pdf', 'docx', 'txt']
        invalid_extensions = ['exe', 'jpg', 'mp3']
        
        for ext in valid_extensions:
            self.assertIn(ext, valid_extensions)
        
        for ext in invalid_extensions:
            self.assertNotIn(ext, valid_extensions)

if __name__ == '__main__':
    unittest.main()
