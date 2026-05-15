import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.text_processor import TextFeatureExtractor
from models.trainer import ModelTrainer

def main():
    data_path = os.path.join(os.path.dirname(__file__), '../data/training_data.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Файл с данными не найден: {data_path}")
        logger.info("Генерирую синтетические данные...")
        from data.generate_samples import generate_training_data
        generate_training_data(5000)
    
    logger.info("Загрузка данных...")
    df = pd.read_csv(data_path)
    
    df['cleaned_text'] = df['text']
    
    logger.info("Кодирование целевой переменной...")
    le = LabelEncoder()
    y = le.fit_transform(df['level'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info("Извлечение признаков...")
    extractor = TextFeatureExtractor(max_features=5000)
    X_train_features = extractor.fit_transform(X_train)
    X_test_features = extractor.transform(X_test)
    
    logger.info("Обучение моделей...")
    trainer = ModelTrainer()
    
    lr_metrics = trainer.train_logistic_regression(
        X_train_features, y_train, X_test_features, y_test
    )
    logger.info(f"Logistic Regression F1: {lr_metrics['f1_score']:.3f}")
    
    cb_metrics = trainer.train_catboost(
        X_train_features, y_train, X_test_features, y_test
    )
    logger.info(f"CatBoost F1: {cb_metrics['f1_score']:.3f}")
    
    models_path = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_path, exist_ok=True)
    
    trainer.save_models(models_path)
    
    joblib.dump(extractor.tfidf, os.path.join(models_path, 'tfidf_vectorizer.joblib'))
    joblib.dump(le, os.path.join(models_path, 'level_encoder.joblib'))
    
    logger.info(f"Модели сохранены в {models_path}/")
    logger.info("Запустите демо: streamlit run src/api/app.py")

if __name__ == "__main__":
    main()
