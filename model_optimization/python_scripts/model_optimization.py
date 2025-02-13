import pandas as pd
import numpy as np
import yaml
import os
import joblib
import sys

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ✅ Force immediate log output (disable buffering)
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# Load YAML config
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Define paths and parameters from config
cleaned_train_data_path = config["paths"]["cleaned_train"]
model_output_path = config["paths"]["model_output"]
target_column = config["parameters"]["target_column"]
test_size = config["parameters"]["test_size"]
random_state = config["parameters"]["random_state"]
fine_tuning_params = config["fine_tuning"]

# ------------------------------------------------------------
# Function 1: Splitting the Data for Training and Testing
# ------------------------------------------------------------

def split_data(data, target_column, test_size, random_state):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    print("✅ Data split into training and testing sets")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ------------------------------------------------------------
# Function 2: Training the Model with Fine-Tuning
# ------------------------------------------------------------

def train_model_with_fine_tuning(X_train, y_train, model_type, params):
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

    models = {
        "logistic_regression": Pipeline([
            ('scaler', StandardScaler()), 
            ('model', LogisticRegression(
                C=params.get("C", 0.5),
                penalty=params.get("penalty", "l1"),
                solver=params.get("solver", "saga"),
                max_iter=params.get("max_iter", 5000),
                class_weight="balanced",
                random_state=random_state,
            ))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 15),
            min_samples_split=params.get("min_samples_split", 10),
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": Pipeline([
            ('scaler', StandardScaler()), 
            ('model', GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 6),
                min_samples_split=params.get("min_samples_split", 10),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=random_state,
            ))
        ])
    }

    if model_type == "voting_ensemble":
        models["voting_ensemble"] = Pipeline([
            ('voting', VotingClassifier(
                estimators=[
                    ('logistic', models["logistic_regression"]),
                    ('random_forest', models["random_forest"]),
                    ('gradient_boosting', models["gradient_boosting"]),
                ],
                voting='soft',
                n_jobs=-1
            ))
        ])

    if model_type not in models:
        raise ValueError("❌ Invalid model_type.")

    model = models[model_type]

    # Perform Cross-Validation
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"{model_type} Cross-Validation Accuracy Scores: {scores}")
    print(f"{model_type} Mean Accuracy: {np.mean(scores):.4f}\n")

    # Train the specified model
    model.fit(X_train, y_train)
    print(f"✅ Model ({model_type}) trained with fine-tuning.")
    return model


# ------------------------------------------------------------
# Function 3: Evaluating trained models
# ------------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    return metrics["F1 Score"]


# ------------------------------------------------------------
# Function 4: Saving the Best Model
# ------------------------------------------------------------

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"✅ Model saved to {output_path}")


# ------------------------------------------------------------
# Model Optimization Pipeline Execution
# ------------------------------------------------------------

def optimization_pipeline():
    if not os.path.exists(cleaned_train_data_path):
        raise FileNotFoundError(f"❌ File not found: {cleaned_train_data_path}")

    print("\n📂 Loading dataset...")
    cleaned_train_data = pd.read_csv(cleaned_train_data_path)
    print("✅ Dataset loaded!")

    X_train, X_test, y_train, y_test = split_data(cleaned_train_data, target_column, test_size, random_state)

    fine_tuned_models = {
        model: train_model_with_fine_tuning(X_train, y_train, model, fine_tuning_params[model])
        for model in fine_tuning_params
    }

    scores = {model: evaluate_model(fine_tuned_models[model], X_test, y_test) for model in fine_tuned_models}
    best_model_name = max(scores, key=scores.get)
    best_model = fine_tuned_models[best_model_name]

    print(f"\n🏆 Best model: {best_model_name} with F1 Score: {scores[best_model_name]:.2f}")
    save_model(best_model, model_output_path)


if __name__ == "__main__":
    optimization_pipeline()
