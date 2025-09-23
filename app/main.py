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

# TODO: mount static file
# from pathlib import Path
# BASE_DIR = Path(__file__).resolve().parent
# app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

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
































# OLD
# --------------------------


# from enum import Enum
#
# class Mode(str, Enum):
#     TRAIN = "train"
#     TEST = "test"

# fake_items_db = [
#     {"item_name": "Foo"},
#     {"item_name": "Bar"},
#     {"item_name": "Baz"}
# ]

# async def read_item(item_id: str, user_id: int, req: str, q: str | None = None, brief: bool = False):
#     item = {"item_id": item_id, "owner_id": user_id, "required": req}
#
#     if q:
#         item.update({"q": q})
#
#     if not brief:
#         item.update({"Description": "This items has a long history of being an item and other itemly things"})
#
#     return item

# @app.post("/items/")
# async def create_item(item: Item):
#     item_dict = item.model_dump()
#
#     if item.tax is not None:
#         price_with_tax = item.price * item.tax
#         item_dict.update({"price_with_tax": price_with_tax})
#
#     return item_dict



# items_db: Dict[int, dict] = {
#     101: {"name": "Laptop", "price": 999.99},
#     102: {"name": "Mouse", "price": 29.99},
# }
#
# # Read a single item
# @app.get("/items/{item_id}")
# async def get_item(item_id: int):
#     if item_id not in items_db:
#         raise HTTPException(status_code=404, detail="Item not found")
#
#     return {
#         "item_id": item_id,
#         "item": items_db[item_id]
#     }
#
# # Read all items in db
# @app.get("/items/")
# async def get_items(item: Item):
#     return {"items": items_db}
#
# # Create new item
# @app.post("/items/{item_id}")
# async def create_item(item: Item):
#     new_id = max(items_db.keys()) + 1 if items_db else 101
#     items_db[new_id] = item.model_dump()
#     return {
#         "item_id": new_id,
#         "item": items_db[new_id]
#     }
#
# # Replace entire item
# @app.put("/items/{item_id}")
# async def update_item(item_id: int, item: Item):
#     if item_id not in items_db:
#         raise HTTPException(status_code=404, detail="Item not found")
#
#     items_db[item_id] = item.model_dump()
#     return {
#         "item_id": item_id,
#         "item": items_db[item_id]
#     }















