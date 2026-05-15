from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.text_processor import TextFeatureExtractor

app = FastAPI(title="CV Analyzer API")

classifier = None
vectorizer = None
encoder = None

@app.on_event("startup")
async def load_models():
    global classifier, vectorizer, encoder
    models_path = os.path.join(os.path.dirname(__file__), '../../models')
    classifier = joblib.load(f"{models_path}/catboost.joblib")
    vectorizer = joblib.load(f"{models_path}/tfidf_vectorizer.joblib")
    encoder = joblib.load(f"{models_path}/level_encoder.joblib")

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    level: str
    confidence: float
    skills: list

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    extractor = TextFeatureExtractor()
    
    features = vectorizer.transform([request.text])
    proba = classifier.predict_proba(features)[0]
    
    level = encoder.inverse_transform([proba.argmax()])[0]
    confidence = float(proba.max())
    skills = extractor.extract_skills_keywords(request.text)
    
    return PredictionResponse(
        level=level,
        confidence=confidence,
        skills=skills
    )

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode('utf-8')
    
    return await predict(TextRequest(text=text))

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": classifier is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
