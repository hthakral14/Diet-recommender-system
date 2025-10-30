🥗 Diet Prediction Web App

A Machine Learning-powered FastAPI application that recommends a personalized diet category — such as Balanced, Low-Carb, or Vegan — based on a user’s demographic and physical attributes.

🚀 Project Overview

The Diet Prediction Web App helps users identify the most suitable diet plan for their fitness goals.
It uses a trained machine learning model to classify users into diet categories based on their age, gender, height, weight, activity level, and goal (e.g., weight loss, maintenance, or gain).

🧠 How It Works

Data Preparation & Feature Engineering:

Collected and cleaned health & nutrition datasets.

Engineered features like BMI (Body Mass Index) to better capture user health metrics.

Model Training:

Trained a classification model to predict diet type.

Evaluated model performance using appropriate metrics.

Serialized the final trained model using Pickle for deployment.

Backend Integration (FastAPI):

Integrated the trained ML model into a FastAPI backend for real-time predictions.

Used Pydantic for input validation to ensure all user data (age, height, weight, etc.) are within valid ranges.

API Functionality:

The app exposes two endpoints:

GET / → Health check endpoint to verify the server is running.

POST /predict_diet → Accepts user data and returns:

Recommended diet category

Calculated BMI

🧩 Example API Usage
🔹 Request
POST /predict_diet
{
  "age": 25,
  "gender": "female",
  "height": 165,
  "weight": 60,
  "activity_level": "moderate",
  "goal": "weight_loss"
}

🔹 Response
{
  "bmi": 22.0,
  "recommended_diet": "Balanced Diet"
}

⚙️ Tech Stack
Component	Technology
Language	Python
Framework	FastAPI
Machine Learning	scikit-learn / pandas / numpy
Model Serialization	Pickle
Validation	Pydantic
Server	Uvicorn
🏗️ Project Structure
diet-prediction-app/
│
├── app.py                 # FastAPI main application
├── model.pkl              # Trained ML model (Pickle file)
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── data/                  # Dataset (optional or example)

💡 Key Features

✅ Real-time personalized diet recommendations
✅ Input validation using Pydantic
✅ BMI calculation and inclusion in prediction
✅ Fast and lightweight REST API
✅ Easily deployable on any cloud platform

🧰 Installation & Setup

Clone the repository:

git clone https://github.com/hthakral14/diet-prediction-app.git
cd diet-prediction-app


Create and activate virtual environment:

python -m venv .venv
source .venv/bin/activate      # For macOS/Linux
.venv\Scripts\activate         # For Windows


Install dependencies:

pip install -r requirements.txt


Run the application:

uvicorn app:app --reload


Test the API:
Open http://127.0.0.1:8000/docs
 to access the interactive Swagger UI.

📈 Future Enhancements

Add user authentication for personalized tracking

Integrate frontend using React or Streamlit

Expand dataset for more accurate predictions

Support for additional diet categories

🧑‍💻 Author

Himanshi Thakral