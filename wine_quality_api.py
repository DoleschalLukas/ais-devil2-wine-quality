from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import pickle

app = FastAPI()

with open("wine_quality_model.pkl", "rb") as f:
    model = pickle.load(f)

from pydantic import BaseModel

class WineQualityRequest(BaseModel):
    fixed_acidity: float
    citric_acid: float
    residual_sugar: float
    volatile_acidity: float
    chlorides: float

class WineQualityRequest(BaseModel):
    fixed_acidity: float = Field(..., description="Fixed Acidity")
    citric_acid: float = Field(..., description="citric acid")
    residual_sugar: float = Field(..., description="residual sugar")
    volatile_acidity: float = Field(..., description="volatile acidity")
    chlorides: float = Field(..., description="chlorides")

class WineQualityResponse(BaseModel):
    quality: float = Field(..., description="Wine quality from 0-9")


# 1. EXACT feature list used during model.fit() in training
TRAINED_FEATURES = ['fixed_acidity', 'citric_acid', 'residual_sugar', 'volatile_acidity', 'chlorides']


@app.post("/predict")
def predict(payload: WineQualityRequest):
    try:
        # Convert incoming data to a raw dictionary
        data_dict = payload.model_dump()

        # If your incoming JSON uses spaces (e.g., "fixed acidity"),
        # map them to underscores to match training keys
        sanitized_dict = {key.replace(" ", "_"): val for key, val in data_dict.items()}

        # Construct DataFrame
        input_df = pd.DataFrame([sanitized_dict])

        # CRITICAL: Force the exact same column ordering as training
        input_df = input_df[TRAINED_FEATURES]

        # Generate prediction
        prediction = model.predict(input_df)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        # This ensures the real error shows up in your terminal logs!
        raise HTTPException(status_code=500, detail=f"Internal Model Error: {str(e)}")

@app.get("/wine-quality", response_model=WineQualityResponse)
def detect_outliers(
    fixed_acidity: float = Query(..., description="Fixed Acidity"),
    citric_acid: float = Query(..., description="citric acid"),
    residual_sugar: float = Query(..., description="residual sugar"),
    volatile_acidity: float = Query(..., description="volatile acidity"),
    chlorides: float = Query(..., description="chlorides")
):
    input_df = pd.DataFrame([{
        "fixed_acidity": fixed_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "volatile_acidity": volatile_acidity,
        "chlorides": chlorides
    }])

    try:
        preds = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    quality = int(preds)
    return WineQualityResponse(quality=quality)