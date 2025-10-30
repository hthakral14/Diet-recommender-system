import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional

# Pydantic Model for User Data (corrected for v2)
class UserData(BaseModel):
    age: int = Field(..., gt=10, description="Age of the user")
    gender: Literal[0, 1] = Field(description="Female=0, male=1")
    weight: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    activity_level: int = Field(..., ge=1, le=5)
    goal: Literal[0, 1, 2] = Field(description="loss=0, maintenance=1, gain=2")
    diet_category: Optional[int] = Field(None, ge=0, le=2)  # Optional
    exercise_type: Optional[int] = Field(None, ge=0, le=2)  # Optional

    # Computed fields
    bmi: float = Field(default=None, description="Calculated BMI")
    verdict: str = Field(default=None, description="Health verdict based on BMI")

    @field_validator('bmi', mode='before')
    @classmethod
    def calculate_bmi(cls, v, info):
        values = info.data
        if 'weight' in values and 'height' in values:
            height_m = values['height'] / 100  # Convert cm to m
            return round(values['weight'] / (height_m ** 2), 2)
        return v

    @field_validator('verdict', mode='before')
    @classmethod
    def calculate_verdict(cls, v, info):
        values = info.data
        if 'bmi' in values:
            bmi = values['bmi']
            if bmi < 18.5:
                return "underweight"
            elif 18.5 <= bmi < 25:
                return "normal"
            elif 25 <= bmi < 30:
                return "overweight"
            else:
                return "obese"
        return v

# Load training data from Pickle
try:
    with open('Train_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded successfully! Shape: {data.shape}")
except FileNotFoundError:
    print("Error: Train_data.pkl not found. Run the data saving script first.")
    exit()

# Augment data with BMI and Verdict using Pydantic
augmented_data = []
for _, row in data.iterrows():
    try:
        user = UserData(**row.to_dict())
        augmented_data.append({
            **row.to_dict(),
            'bmi': user.bmi,
            'verdict': user.verdict
        })
    except Exception as e:
        print(f"Error processing row: {e}. Skipping.")

data = pd.DataFrame(augmented_data)
print("Data augmented with BMI and Verdict. Sample:")
print(data[['age', 'weight', 'height', 'bmi', 'verdict']].head())

# Prepare features and labels (train on diet_category; adjust for exercise_type if needed)
X = data[['age', 'gender', 'weight', 'height', 'activity_level', 'goal', 'bmi']]
y = data['diet_category']  # Single label; for multi-label, use [data['diet_category'], data['exercise_type']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Save the trained model to Pickle (correct file name)
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Trained model saved to trained_model.pkl")