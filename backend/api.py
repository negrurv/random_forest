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
    print("--- CLOUD DIAGNOSTICS ---")
    
    data_dir = "/app/data"
    if not os.path.exists(data_dir):
        # Fallback for local Mac testing
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print(f"Checking for data in: {data_dir}")
    
    # List EVERYTHING in that folder so we can see it in the logs
    try:
        files = os.listdir(data_dir)
        print(f"Files found in data folder: {files}")
        
        # Automatically find any CSV file in that folder
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        csv_path = os.path.join(data_dir, csv_files[0])
        print(f"✅ Selecting CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        raise e


class PredictRequest(BaseModel):
    samples: List[List[float]] 

@app.post("/predict")
def predict(request: PredictRequest):
    flat_features = [value for sample in request.samples for value in sample]
    num_samples_to_predict = len(request.samples)
    
    predictions = rf_model.predict_batch(flat_features, num_samples_to_predict, NUM_FEATURES)
    
    return {"predictions": predictions}