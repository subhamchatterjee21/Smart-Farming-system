import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the trained models
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("rain_model.pkl", "rb") as rain_model_file:
    rain_model = pickle.load(rain_model_file)

# Define the input model using Pydantic
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float

# Define the output model using Pydantic
class PredictionResponse(BaseModel):
    predicted_crop: str
    required_rain: float

# Create FastAPI application
app = FastAPI()

# Create the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_crop(input_data: CropInput):
    # Convert the input data to a numpy array
    input_array = np.array([[input_data.N, input_data.P, input_data.K, input_data.temperature, input_data.humidity, input_data.ph]])
    
    # Predict the crop label
    predicted_crop = model.predict(input_array)[0]
    
    # Predict the required rain
    required_rain = rain_model.predict(input_array)[0]
    
    # Return the predictions
    return PredictionResponse(predicted_crop=predicted_crop, required_rain=required_rain)


# Run the application
# This will be done from the command line, e.g., `uvicorn app:app --reload`
