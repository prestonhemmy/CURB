#!/bin/bash

source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000