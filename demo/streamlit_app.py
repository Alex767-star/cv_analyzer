import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from api.app import CVAnalyzerApp

st.set_page_config(
    page_title="CV Analyzer | ML Production",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("Настройки")
    model_choice = st.selectbox("Модель", ["CatBoost", "BERT", "Logistic Regression"])
    
    st.markdown("---")
    st.markdown("### Эксперименты")
    
    import pandas as pd
    experiments = pd.DataFrame({
        'Run': ['catboost_v2', 'bert_finetuned', 'baseline_lr'],
        'F1': [0.87, 0.91, 0.78],
        'Status': ['✅ Production', '🔄 Testing', '✅ Archived']
    })
    st.dataframe(experiments, hide_index=True)
    
    st.markdown("---")
    st.markdown("### MLflow Dashboard")
    st.markdown("[Открыть MLflow](http://localhost:5000)")
    
    st.markdown("---")
    st.markdown("### Информация")
    st.markdown("**Автор:** Alex767-star")
    st.markdown("**Версия:** 1.0.0")
    st.markdown(f"**Модель:** {model_choice}")

if __name__ == "__main__":
    models_path = os.path.join(os.path.dirname(__file__), '../models')
    app = CVAnalyzerApp(models_path)
    app.run()
