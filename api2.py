from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")


# Load exported TensorFlow model
model = tf.keras.layers.TFSMLayer(
    r"C:\Users\niniz\best_heart.tf",
    call_endpoint="serve"
)


class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.post("/predict")
def predict(data: HeartInput):

    features = np.array([[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]], dtype=np.float32)

    # Run model
    prediction = model(features)

    # Handle TensorFlow/Keras output
    if isinstance(prediction, dict):
        prediction = next(iter(prediction.values()))

    prediction_value = float(np.asarray(prediction).reshape(-1)[0])

    result = int(prediction_value > 0.5)

    return {
        "prediction_probability": prediction_value,
        "heart_disease": result,
        "message": (
            "Heart Disease Detected"
            if result == 1
            else "No heart disease"
        )
    }