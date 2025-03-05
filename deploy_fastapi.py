from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Union, List
import joblib
import pandas as pd
import os


class SampleData(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]


models = {}
model_files = {
    "SVM": os.path.join("models", "SVM_model.pkl"),
    "KNN": os.path.join("models", "KNN_model.pkl"),
    "AdaBoost": os.path.join("models", "AdaBoost_model.pkl"),
    "NaiveBayes": os.path.join("models", "NaiveBayes_model.pkl"),
    "XGBoost": os.path.join("models", "XGBoost_model.pkl"),
}
for name, filepath in model_files.items():
    models[name] = joblib.load(filepath)

# Creating the FastAPI App.
app = FastAPI()


@app.post("/predict")
def predict(sample_data: SampleData):
    try:
        # Convert input data to a DataFrame
        sample = sample_data.data
        df = pd.DataFrame(sample if isinstance(sample, list) else [sample])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {e}")

    # Make predictions using the loaded models
    predictions = {}
    for name, model in models.items():
        preds = model.predict(df)
        predictions[name] = preds.tolist()

    return {"predictions": predictions}


# To run: uvicorn deploy_fastapi:app --reload

# To access the Swagger UI: http://127.0.0.1:8000/docs

# For Postman: http://127.0.0.1:8000/predict
