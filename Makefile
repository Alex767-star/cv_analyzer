.PHONY: install generate train run api bot docker clean

install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

generate:
	. venv/bin/activate && python src/data/generate_samples.py

train:
	. venv/bin/activate && python src/train_pipeline.py

run:
	. venv/bin/activate && streamlit run src/api/app.py

api:
	. venv/bin/activate && uvicorn src.api.fastapi_app:app --reload

bot:
	. venv/bin/activate && python src/bot/tg_bot.py

docker:
	docker-compose up --build

clean:
	rm -rf models/*.joblib data/*.csv mlruns/ .streamlit/
	find . -type d -name __pycache__ -exec rm -rf {} +
