from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Dict

from src.config import *
from src.predict import classifier_service

import time


# load saved model (once) when server starts
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # On Startup:

    print ("\n" + "=" * 50)
    print("Starting server...\nLoading model...")
    print("Estimated wait time: 3 seconds.")
    print("\n" + "=" * 50)

    success = classifier_service.load_model(MODEL_PATH)

    print()

    if success:
        print(f"Model loaded in {classifier_service.model:.2f} seconds.")
        print("You may now access the API at http://127.0.0.1:8000")
        print("See interactive docs at http://127.0.0.1:8000/docs")

    else:
        print(f"Failed to load model: {classifier_service.model_load_error}")

    print()

    yield   # transfer control to app

    # On Shutdown:

    # clean up
    print("\n" + "=" * 50)
    print("Shutting down server...")
    print("Cleaning up resources...")
    print("\n" + "=" * 50)

app = FastAPI(
    title="News Classifier",
    description="BERT-based news classification app",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# request / response models
class TextRequest(BaseModel):
    text: str

    @field_validator('text')
    def text_must_not_be_empty(cls, value):
        if not value:
            raise ValueError("'text' must not be empty")

        if len(value) > 10000:
            raise ValueError("'text' is too long (enter a max of 10 000 characters)")

        return value

class PredictionResponse(BaseModel):
    text: str
    category: str
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time: float

# endpoints
@app.get("/")
async def root():
    """
    Root endpoint

    Shows API information
    """

    return {
        "message": "News Classifier API",
        "status": "MODEL LOADED" if classifier_service.is_loaded else "MODEL NOT LOADED",
        "model_loaded": classifier_service.model_loaded,
        "model_load_time": classifier_service.model_load_time,
        "endpoints": {
            "GET /": "This message",
            "GET /health": "Diagnostic health summary",
            "POST /predict": "Classify a news article text",
        }
    }

@app.get("/health")
async def health():
    """
    Health check endpoint

    Shows model loading diagnostic description
    """

    return {
        "status": "healthy" if classifier_service.is_loaded else "unhealthy",
        "model_loaded": classifier_service.is_loaded,
        "model_load_time": classifier_service.model_load_time,
        "error": classifier_service.model_load_error,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """
    Classify news text into categories

    Sends a POST request with JSON body of the form:
        {"text": "user-entered news article text to classify"}
    """

    # model availability error handling
    if not classifier_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not yet loaded. Try again later.")

    start = time.time()

    # run prediction
    try:
        result = classifier_service.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        text=result["input_text"],
        category=result["predicted_class"],
        confidence=result["confidence"],
        all_probabilities=result["all_probabilities"],
        processing_time=time.time() - start,
    )

# TODO: Add support for user batch input (capped at 10)
#  @app.post("/predict-batch")