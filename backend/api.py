# backend/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
import rf_cpp

app = FastAPI(title="Football Predictor API")

# --- THIS MUST BE RIGHT HERE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], # This is what allows the 'OPTIONS' method!
    allow_headers=["*"],
)
# -------------------------------

rf_model = None
NUM_FEATURES = 4

@app.on_event("startup")
def startup_event():
    global rf_model
    print("Loading football data...")
    
    # 1. Check common locations for the CSV
    possible_paths = [
        "/app/data/clean_football_data.csv",                     # Docker Absolute
        "./data/clean_football_data.csv",                      # Local relative from root
        "../data/clean_football_data.csv",                     # Local relative from backend folder
        os.path.join(os.path.dirname(__file__), "..", "data", "clean_football_data.csv") # Dynamic Mac path
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"✅ Found data at: {csv_path}")
            break
            
    if csv_path is None:
        print("❌ CRITICAL ERROR: Could not find clean_football_data.csv in any location!")
        # Printing current directory to help debug logs if it fails
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of /app: {os.listdir('/app') if os.path.exists('/app') else 'not found'}")
        raise FileNotFoundError("CSV data file missing from container.")

    # 2. Load the data
    df = pd.read_csv(csv_path)

class PredictRequest(BaseModel):
    samples: List[List[float]] 

@app.post("/predict")
def predict(request: PredictRequest):
    flat_features = [value for sample in request.samples for value in sample]
    num_samples_to_predict = len(request.samples)
    
    predictions = rf_model.predict_batch(flat_features, num_samples_to_predict, NUM_FEATURES)
    
    return {"predictions": predictions}