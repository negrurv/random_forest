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
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    try:
        files = os.listdir(data_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
            
        csv_path = os.path.join(data_dir, csv_files[0])
        print(f"✅ Selecting CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # --- THE MISSING TRAINING LOGIC IS BACK ---
        print("Preparing data for C++ Engine...")
        df_numeric = df.select_dtypes(include=['number']).dropna()
        
        if 'Target' not in df_numeric.columns:
            raise ValueError(f"'Target' column missing! Available: {df_numeric.columns.tolist()}")
            
        X = df_numeric.drop(columns=['Target'])
        y = df_numeric['Target']
        
        X_flat = X.values.flatten().tolist()
        y_list = y.tolist()
        
        print(f"Training C++ Random Forest on {len(y)} samples...")
        
        # Initialize the model (using 10 trees, max depth 5 - adjust if you had different numbers!)
        rf_model = rf_cpp.RandomForest(10, 5) 
        rf_model.fit(X_flat, y_list, len(y), NUM_FEATURES)
        
        print("✅ Model trained and ready to predict!")
        # ------------------------------------------

    except Exception as e:
        print(f"❌ Failed to start: {e}")
        raise e


class PredictRequest(BaseModel):
    samples: List[List[float]] 

@app.post("/predict")
def predict(request: PredictRequest):
    flat_features = [value for sample in request.samples for value in sample]
    num_samples_to_predict = len(request.samples)
    
    predictions = rf_model.predict_batch(flat_features, num_samples_to_predict, NUM_FEATURES)
    
    return {"predictions": predictions}