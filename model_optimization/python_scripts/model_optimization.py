import pandas as pd
import numpy as np
import yaml
import os
import joblib
import sys
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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

def train_model_with_fine_tuning(X_train, y_train, config):
    cv_params = config["cross_validation"]

    logistic_regression_params = config["logistic_regression"]
    random_forest_params = config["random_forest"]
    gradient_boosting_params = config["gradient_boosting"]

    kfold = StratifiedKFold(n_splits=cv_params['n_splits'], shuffle=cv_params['shuffle'], random_state=cv_params['random_state'])

    base_models = {
        "logistic_regression": (LogisticRegression, logistic_regression_params),
        "random_forest": (RandomForestClassifier, random_forest_params),
        "gradient_boosting": (GradientBoostingClassifier, gradient_boosting_params)
    }

    best_models = {}

    for model_name, (model_class, param_grid) in base_models.items():
        param_combinations = [dict(zip(param_grid.keys(), values)) for values in zip(*param_grid.values())]

        best_score = 0
        best_model = None

        for params in param_combinations:
            model = model_class(**params)
            scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='f1', n_jobs=-1)
            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_model = model

        best_model.fit(X_train, y_train)
        best_models[model_name] = best_model
        print(f"✅ Best {model_name}: F1 Score = {best_score:.4f}")

    voting_ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft',
        n_jobs=-1
    )

    voting_ensemble.fit(X_train, y_train)
    print("✅ Voting ensemble trained with best models.")
    return voting_ensemble

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

    # Call the train_model_with_fine_tuning function from external module
    fine_tuned_model = train_model_with_fine_tuning(X_train, y_train, config)

    # Evaluate the trained model
    f1_score = evaluate_model(fine_tuned_model, X_test, y_test)

    print(f"\n🏆 Best model trained with an F1 Score of {f1_score:.2f}")
    save_model(fine_tuned_model, model_output_path)


if __name__ == "__main__":
    optimization_pipeline()