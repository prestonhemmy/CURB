from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, AsyncExitStack

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from typing import Dict, Optional
import torch
from sentry_sdk.utils import to_string
from starlette.responses import HTMLResponse
from transformers import BertTokenizer

from src import create_model
from src.model import TopicClassifier
from src.config import *

import time

from src.predict import classifier_service


# load saved model (once) when server starts
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # on startup:

    global model_load_time, model_load_error

    print ("\n" + "=" * 50)
    print("Starting server...\nLoading model...")
    print("Estimated wait time: 3 seconds.")
    print("\n" + "=" * 50)

    start = time.time()
    success = classifier_service.load_model(MODEL_PATH)
    model_load_time = time.time() - start

    print()

    if success:
        print(f"Model loaded in {model_load_time:.2f} seconds.")
        print("You may now access the API at http://127.0.0.1:8000")
        print("See interactive docs at http://127.0.0.1:8000/docs")

    else:
        model_load_error = "Failed to load model (see logs)."
        print(f"Error: {model_load_error}")

    print()

    yield   # transfer control to app

    # on shutdown:

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

# TODO: Add CORS middleware (for frontend integration)

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

# globals
model_load_time : Optional[float] = None
model_load_error: Optional[str] = None

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
        # TODO: model_loaded,
        #       model_load_time,
        #       instructions (Optional)
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
        "model_loaded": classifier_service.is_loaded
        # TODO: model_load_time, error, available_classes (Optional)
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

    # TODO: init prediction time tracker

    # run prediction
    result = classifier_service.predict(request.text)

    # TODO: generic error handling

    # TODO: add processing time to response

    return result

# TODO: Add support for user batch input (capped at 10)
#  @app.post("/predict-batch")