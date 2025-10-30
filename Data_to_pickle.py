import pandas as pd
import pickle

try:
    # Load the CSV (ensure Dataset.csv exists and has data)
    Data = pd.read_csv("Dataset.csv")
    
    # Check if data is loaded (optional debug)
    print(f"Data loaded successfully! Shape: {Data.shape}")
    
    # Save to Pickle (correct filename)
    with open('Train_data.pkl', 'wb') as f:
        pickle.dump(Data, f)
    
    print("Training data saved to Train_data.pkl")
except FileNotFoundError:
    print("Error: Dataset.csv not found. Please ensure the file exists in the same directory.")
except pd.errors.EmptyDataError:
    print("Error: Dataset.csv is empty or has no columns. Add data to the file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")