import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVDataset:
    def __init__(self, data_path: str, target_cols: List[str]):
        self.data_path = data_path
        self.target_cols = target_cols
        self.df = None
        self.label_encoders = {}
        
    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} samples with columns: {self.df.columns.tolist()}")
        return self.df
    
    def clean_text(self, text: str) -> str:
        import re
        text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', str(text))
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    def preprocess_targets(self) -> Dict[str, Dict]:
        from sklearn.preprocessing import LabelEncoder
        
        for col in self.target_cols:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('unknown'))
            self.label_encoders[col] = le
            
        return self.label_encoders
    
    def stratified_split(self, target_col: str, n_splits: int = 5) -> Tuple:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = self.df['cleaned_text'].values
        y = self.df[f'{target_col}_encoded'].values
        
        for train_idx, val_idx in skf.split(X, y):
            yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]
