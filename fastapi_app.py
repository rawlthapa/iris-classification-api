from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib

# Initialize FastAPI app
app=FastAPI()

# Load the trained model
# Load the trained model
try:
    model = joblib.load('iris_logistic_regression.pkl')
except FileNotFoundError:
    raise RuntimeError("Model file not found. Ensure 'iris_logistic_regression.pkl' is in the correct directory.")

# Define request model
class Features(BaseModel):
     features:conlist(float, min_length=4, max_length=4)  #Ensure the features list has exactly 4 float values 


# # Define a simple route to check if the API is running
# @app.get("/")
# def root():
#     return {"message": "API is running"}

# Define a route for the prediction API
@app.post('/predict')
def predict(data: Features):
     try:
        #   Predict using model
        prediction=model.predict([data.features])
        return {'prediction': int(prediction[0])}
        
     except Exception as e:
        #  Handles error during prediction
        raise HTTPException(status_code=500, detail=str(e))