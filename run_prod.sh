#!/bin/bash

source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Note: each worker loads its own model copy (~413MB)