import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.text_processor import TextFeatureExtractor

class CVAnalyzerBot:
    def __init__(self, token: str, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../../models')
        
        self.token = token
        self.model = joblib.load(f"{model_path}/catboost.joblib")
        self.vectorizer = joblib.load(f"{model_path}/tfidf_vectorizer.joblib")
        self.encoder = joblib.load(f"{model_path}/level_encoder.joblib")
        self.extractor = TextFeatureExtractor()
    
    async def start(self, update: Update, context):
        await update.message.reply_text(
            "👋 Привет! Я анализирую резюме и определяю уровень специалиста.\n"
            "Просто отправь мне текст резюме!"
        )
    
    async def analyze(self, update: Update, context):
        text = update.message.text
        
        features = self.vectorizer.transform([text])
        proba = self.model.predict_proba(features)[0]
        
        level = self.encoder.inverse_transform([proba.argmax()])[0]
        confidence = proba.max()
        
        skills = self.extractor.extract_skills_keywords(text)
        
        response = (
            f"🎯 Уровень: {level.upper()}\n"
            f"📊 Уверенность: {confidence:.1%}\n"
        )
        
        if skills:
            response += f"🛠 Навыки: {', '.join(skills[:5])}"
        
        await update.message.reply_text(response)
    
    def run(self):
        app = Application.builder().token(self.token).build()
        
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.analyze))
        
        print("Бот запущен...")
        app.run_polling()

if __name__ == "__main__":
    token = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
    bot = CVAnalyzerBot(token)
    bot.run()
