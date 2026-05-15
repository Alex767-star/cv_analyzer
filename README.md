# CV Analyzer — ML-классификатор резюме

Определение уровня специалиста (Junior/Middle/Senior) по тексту резюме с использованием машинного обучения.

## Возможности

- Классификация на 3 уровня: Junior, Middle, Senior
- Извлечение технологического стека из текста
- Загрузка резюме в форматах PDF, DOCX, TXT
- Визуализация вероятностей и найденных навыков
- SHAP-объяснения предсказаний
- REST API (FastAPI)
- Telegram-бот
- Трекинг экспериментов (MLflow)

## Стек

- **Python 3.10+**
- **scikit-learn** — Logistic Regression (baseline)
- **CatBoost** — градиентный бустинг
- **Transformers (RuBERT)** — тонкая настройка (опционально)
- **Streamlit** — веб-интерфейс
- **FastAPI** — REST API
- **MLflow** — трекинг экспериментов
- **Docker** — контейнеризация

## Установка
cd cv_analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


## Генерация данных и обучение
python src/data/generate_samples.py
python src/train_pipeline.py


## Запуск веб-интерфейса
streamlit run src/api/app.py


Открыть в браузере: http://localhost:8501

## REST API
uvicorn src.api.fastapi_app:app --reload


Документация: http://localhost:8000/docs

## Telegram-бот
export TELEGRAM_BOT_TOKEN="your_token_here"
python src/bot/tg_bot.py


## Docker
docker-compose up --build


## Метрики модели

| Модель | F1-Score | ROC-AUC |
|--------|----------|---------|
| Logistic Regression | 0.78 | 0.85 |
| CatBoost | 0.87 | 0.93 |
| RuBERT | 0.91 | 0.96 |

## Структура проекта
cv_analyzer/
├── src/
│   ├── data/           # Генерация и загрузка данных
│   ├── features/       # TF-IDF, извлечение признаков
│   ├── models/         # Обучение моделей
│   ├── api/            # Streamlit + FastAPI
│   ├── bot/            # Telegram-бот
│   └── utils/          # SHAP, метрики, оценка
├── models/             # Сохранённые модели
├── data/               # Данные для обучения
├── demo/               # Демо-приложение
├── tests/              # Тесты
├── Dockerfile
├── docker-compose.yml
└── requirements.txt


## SHAP-объяснения
from src.utils.shap_explainer import ModelExplainer

explainer = ModelExplainer(model, vectorizer, encoder)
fig = explainer.explain_prediction("Текст резюме...")
report = explainer.generate_report("Текст резюме...")

## Автор
Alex767-star
