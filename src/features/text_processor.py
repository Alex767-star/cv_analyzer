import re
from typing import List, Optional
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class TextFeatureExtractor:
    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            analyzer='char_wb'
        )
        
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('russian'))
        except:
            self.stop_words = set()
    
    def lemmatize(self, text: str) -> str:
        words = text.lower().split()
        return ' '.join(words)
    
    def remove_stopwords(self, text: str) -> str:
        if not self.stop_words:
            return text
        words = [w for w in text.split() if w not in self.stop_words]
        return ' '.join(words)
    
    def extract_skills_keywords(self, text: str) -> List[str]:
        tech_keywords = [
            'python', 'java', 'javascript', 'sql', 'docker', 'kubernetes',
            'aws', 'gcp', 'azure', 'tensorflow', 'pytorch', 'spark',
            'hadoop', 'kafka', 'redis', 'postgresql', 'mongodb', 'linux',
            'git', 'ci/cd', 'ansible', 'terraform', 'prometheus', 'grafana'
        ]
        found = [kw for kw in tech_keywords if kw in text.lower()]
        return list(set(found))
    
    def fit_transform(self, texts: List[str]) -> csr_matrix:
        processed = [self.remove_stopwords(self.lemmatize(str(text))) for text in texts]
        return self.tfidf.fit_transform(processed)
    
    def transform(self, texts: List[str]) -> csr_matrix:
        processed = [self.remove_stopwords(self.lemmatize(str(text))) for text in texts]
        return self.tfidf.transform(processed)
