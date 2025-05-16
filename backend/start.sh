#!/bin/bash

# Start FastAPI app and serve frontend
uvicorn backend.main:app --host 0.0.0.0 --port 10000 --reload