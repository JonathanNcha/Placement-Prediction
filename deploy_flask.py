from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load trained models
models = {}
model_files = {
    "SVM": os.path.join("models", "SVM_model.pkl"),
    "KNN": os.path.join("models", "KNN_model.pkl"),
    "AdaBoost": os.path.join("models", "AdaBoost_model.pkl"),
    "NaiveBayes": os.path.join("models", "NaiveBayes_model.pkl"),
    "XGBoost": os.path.join("models", "XGBoost_model.pkl"),
}

# Loops over each model name and file path.
for name, filepath in model_files.items():
    # Loads the model file using joblib and stores it in the models dictionary under its name.
    models[name] = joblib.load(filepath)

# Initializes a new Flask web application.
app = Flask(__name__)


# Define prediction endpoint. Specifies that the function predict() should be called when a POST request is sent to the "/predict" URL.
@app.route("/predict", methods=["POST"])
def predict():
    # Reads the JSON body of the incoming request. Checks ensure that input exists and contains a key "data". If missing, it returns an error JSON with a 400 status code.
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "No input data provided"}), 400
    # Retrieves the value associated with the "data" key.
    sample = input_data.get("data")
    if sample is None:
        return jsonify({"error": "Key 'data' not found in JSON"}), 400

    try:
        # Converts the input into a pandas DataFrame. If the provided data is a single record, it wraps it in a list.
        df = pd.DataFrame(sample if isinstance(sample, list) else [sample])
    except Exception as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400

    # The code then loops over each loaded model, calls the predict() method of each model using the DataFrame df, converts predictions to a list, and stores them in a predictions dictionary.
    predictions = {}
    for name, model in models.items():
        preds = model.predict(df)
        predictions[name] = preds.tolist()  # Convert numpy array to list

    # Finally, return jsonify({"predictions": predictions}) sends back a JSON response with the predictions.
    return jsonify({"predictions": predictions})


# Checks whether this script is executed directly.
if __name__ == "__main__":
    # Starts the Flask server on port 5001, accessible on all network interfaces, with debug mode enabled.
    app.run(host="0.0.0.0", port=5001, debug=True)
