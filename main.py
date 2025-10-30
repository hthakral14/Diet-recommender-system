from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pickle
import numpy as np

app = FastAPI(title="Diet Prediction App", version="1.0.0")

# Pydantic model for user input validation
class UserData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal[0, 1]  # 0=female, 1=male
    weight: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    activity_level: int = Field(..., ge=1, le=5)
    goal: Literal[0, 1, 2]  # 0=loss, 1=maintenance, 2=gain

# Load the trained model from Pickle (ensure 'trained_model.pkl' exists)
try:
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Trained model file 'trained_model.pkl' not found. Train and save the model first.")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Diet Prediction API is running!"}

@app.post("/predict_diet")
def predict_diet(user_data: UserData):  # Use Pydantic for validation
    # Calculate BMI for features (match training)
    height_m = user_data.height / 100
    bmi = round(user_data.weight / (height_m ** 2), 2)
    
    # Extract features (now includes BMI to match model)
    features = np.array([[user_data.age, user_data.gender, user_data.weight, 
                          user_data.height, user_data.activity_level, user_data.goal, bmi]])
    
    # Predict using the loaded model
    try:
        prediction = model.predict(features)[0]  # e.g., 0=balanced, 1=low-carb
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    # Map to readable output
    diet_map = {0: "balanced", 1: "low-carb", 2: "vegan"}
    return {"diet_category": diet_map.get(prediction, "unknown"), "bmi": bmi}