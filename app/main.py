import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.json")

app = FastAPI()


class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island


def load_model():
    logger.info("Attempting to load model from %s", MODEL_PATH)
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.exception("Failed to load model.")
        raise e

model = load_model()


def preprocess_features(features: PenguinFeatures):
    logger.info("Preprocessing features: %s", features)
    try:
        input_dict = features.model_dump()
        X_input = pd.DataFrame([input_dict]) 
        X_input = pd.get_dummies(X_input, columns=["sex", "island"]) 
        expected_cols = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex_Female",
            "sex_Male",
            "island_Biscoe",
            "island_Dream",
            "island_Torgersen",
        ]
        X_input = X_input.reindex(columns=expected_cols, fill_value=0)
        X_input = X_input.astype(float)
        logger.info("Feature preprocessing completed successfully.")
        return X_input
    except Exception as e:
        logger.exception("Error during preprocessing.")
        raise e


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(features: PenguinFeatures):
    logger.info("Received prediction request.")
    try:
        X_input = preprocess_features(features)
        pred = model.predict(X_input.values)
        logger.info("Prediction successful. Result: %d", int(pred[0]))
        return {"prediction": int(pred[0])}
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction error")


@app.get("/health")
async def health():
    return {"status": "ok"}
