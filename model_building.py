import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import joblib

# Load the preprocessed data
df = pd.read_csv("data/processed_data.csv")

# Split the data into features and target
X = df.drop("PlacementStatus", axis=1)
y = df["PlacementStatus"]

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=15
)


# Function to train models
def train_models(X_train, y_train):
    models = {}

    # Support Vector Machine (SVM)
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    models["SVM"] = svm

    # K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models["KNN"] = knn

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models["NaiveBayes"] = nb

    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train, y_train)
    models["AdaBoost"] = ada

    return models


# Train the models
models = train_models(X_train, y_train)


# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    if not os.path.exists("model_reports"):
        os.makedirs("model_reports")
    if not os.path.exists("models"):
        os.makedirs("models")
    reports = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        reports[name] = {"report": report, "confusion_matrix": cm}

        # Save the classification report
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f"model_reports/{name}_classification_report.csv", index=True)

        # Save the confusion matrix
        df_cm = pd.DataFrame(
            cm,
            index=["Actual_NotPlaced", "Actual_Placed"],
            columns=["Predicted_NotPlaced", "Predicted_Placed"],
        )
        df_cm.to_csv(f"model_reports/{name}_confusion_matrix.csv", index=True)

        # Save the model
        joblib.dump(model, f"models/{name}_model.pkl")

    return reports


# Evaluate the models
reports = evaluate_models(models, X_test, y_test)
