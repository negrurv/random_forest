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
    print("Loading football data and initializing C++ Model...")
    
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "clean_football_data.csv")
    df = pd.read_csv(csv_path)
    
    df_numeric = df.select_dtypes(include=['number']).dropna()
    X = df_numeric.drop(columns=['Target'])
    y = df_numeric['Target']
    
    X_flat = X.values.flatten().tolist()
    y_list = y.values.tolist()
    num_samples = len(y_list)
    
    rf_model = rf_cpp.RandomForest(100, 10, 5, 0.8)
    rf_model.train(X_flat, y_list, num_samples, NUM_FEATURES)
    print("Model is trained and ready to predict matches!")

class PredictRequest(BaseModel):
    samples: List[List[float]] 

@app.post("/predict")
def predict(request: PredictRequest):
    flat_features = [value for sample in request.samples for value in sample]
    num_samples_to_predict = len(request.samples)
    
    predictions = rf_model.predict_batch(flat_features, num_samples_to_predict, NUM_FEATURES)
    
    return {"predictions": predictions}