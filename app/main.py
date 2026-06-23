## basic API code

from fastapi import FastAPI
import pickle
import numpy as np

# create app
app = FastAPI()

#load model
model = pickle.load(open("model.pkl", "rb"))

# home endpoint

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

# prediction endpoint
@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    
    return {"fraud_prediction": int(prediction[0])}
