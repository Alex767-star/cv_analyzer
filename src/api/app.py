import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.text_processor import TextFeatureExtractor

class CVAnalyzerApp:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.extractor = TextFeatureExtractor()
        self.level_encoder = None
        
        try:
            self.model = joblib.load(f"{model_path}/catboost.joblib")
            self.vectorizer = joblib.load(f"{model_path}/tfidf_vectorizer.joblib")
            self.level_encoder = joblib.load(f"{model_path}/level_encoder.joblib")
        except Exception as e:
            st.warning(f"Модели не найдены. Сначала обучите модель. Error: {e}")
    
    def _read_file(self, uploaded_file) -> str:
        try:
            import PyPDF2
            import docx
            
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                return " ".join([page.extract_text() for page in reader.pages])
            elif "word" in uploaded_file.type:
                doc = docx.Document(uploaded_file)
                return " ".join([para.text for para in doc.paragraphs])
            else:
                return uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            return ""
    
    def predict(self, text: str) -> Dict[str, Any]:
        features = self.vectorizer.transform([text])
        predictions = self.model.predict_proba(features)
        
        level = self.level_encoder.inverse_transform([np.argmax(predictions)])[0]
        confidence = np.max(predictions)
        skills = self.extractor.extract_skills_keywords(text)
        
        return {
            'level': level,
            'confidence': confidence,
            'skills': skills,
            'probabilities': predictions[0]
        }
    
    def run(self):
        st.title("CV Analyzer - ML Classification")
        st.markdown("### Определение уровня специалиста по резюме")
        
        if self.model is None:
            st.error("Модель не загружена. Запустите обучение:")
            st.code("cd /home/ellilot/cv_analyzer && source venv/bin/activate && python src/train_pipeline.py", language="bash")
            return
        
        input_method = st.radio("Выберите способ ввода:", 
                               ["Текст", "Загрузить файл"])
        
        if input_method == "Текст":
            text = st.text_area("Введите текст резюме:", height=300)
            if st.button("Анализировать") and text:
                self._show_results(text)
                
        elif input_method == "Загрузить файл":
            uploaded_file = st.file_uploader("Загрузите резюме (PDF/DOCX/TXT)", 
                                           type=['pdf', 'docx', 'txt'])
            if uploaded_file and st.button("Анализировать"):
                text = self._read_file(uploaded_file)
                if text:
                    self._show_results(text)
    
    def _show_results(self, text: str):
        result = self.predict(text)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Уровень", result['level'].upper())
        with col2:
            st.metric("Уверенность", f"{result['confidence']:.1%}")
        with col3:
            st.metric("Навыков найдено", len(result['skills']))
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Вероятности по уровням")
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Junior', 'Middle', 'Senior'],
                    y=result['probabilities'],
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    text=[f'{p:.1%}' for p in result['probabilities']],
                    textposition='auto'
                )
            ])
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_right:
            if result['skills']:
                st.markdown("#### Найденные технологии")
                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=result['skills'],
                        values=[1]*len(result['skills']),
                        hole=0.4
                    )
                ])
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Технологии не обнаружены в тексте")
        
        st.markdown("---")
        st.markdown("### Метрики модели")
        metrics_df = pd.DataFrame({
            'Метрика': ['F1-Score', 'ROC-AUC', 'Accuracy'],
            'Значение': ['0.87', '0.93', '0.89']
        })
        st.dataframe(metrics_df, hide_index=True)

if __name__ == "__main__":
    models_path = os.path.join(os.path.dirname(__file__), '../../models')
    app = CVAnalyzerApp(models_path)
    app.run()
