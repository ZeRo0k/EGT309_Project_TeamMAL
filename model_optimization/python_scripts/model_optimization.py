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

def train_model_with_fine_tuning(X_train, y_train, config_path='config.yaml'):
    # Load hyperparameters from config.yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_type = config.get('model_type')
    if not model_type:
        raise ValueError("❌ No model type specified in config.yaml")

    if model_type not in config:
        raise ValueError(f"❌ Hyperparameters for {model_type} not found in config.yaml")

    param_grid = config[model_type]
    cv_config = config.get('cross_validation')

    kfold = StratifiedKFold(n_splits=cv_config['n_splits'], shuffle=cv_config['shuffle'], random_state=cv_config['random_state'])

    models = {
        "logistic_regression": lambda p: Pipeline([
            ('scaler', StandardScaler()), 
            ('model', LogisticRegression(
                C=p["C"],
                penalty=p["penalty"],
                solver=p["solver"],
                max_iter=p["max_iter"],
                class_weight="balanced",
                random_state=cv_config['random_state'],
            ))
        ]),
        "random_forest": lambda p: RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            n_jobs=-1,
            random_state=cv_config['random_state'],
        ),
        "gradient_boosting": lambda p: Pipeline([
            ('scaler', StandardScaler()), 
            ('model', GradientBoostingClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                min_samples_split=p["min_samples_split"],
                learning_rate=p["learning_rate"],
                random_state=cv_config['random_state'],
            ))
        ])
    }

    if model_type == "voting_ensemble":
        models["voting_ensemble"] = lambda p: VotingClassifier(
            estimators=[
                ('logistic', models["logistic_regression"](p)),
                ('random_forest', models["random_forest"](p)),
                ('gradient_boosting', models["gradient_boosting"](p)),
            ],
            voting='soft',
            n_jobs=-1
        )

    if model_type not in models:
        raise ValueError("❌ Invalid model_type.")

    # Generate parameter combinations
    param_keys = list(param_grid.keys())
    param_combinations = []
    for values in zip(*param_grid.values()):
        param_combinations.append(dict(zip(param_keys, values)))

    best_score = 0
    best_params = None
    best_model = None

    for params in param_combinations:
        print(f"Training {model_type} with params: {params}")
        model = models[model_type](params)

        scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=-1)
        mean_score = np.mean(scores)
        print(f"Cross-Validation Accuracy: {scores} | Mean Accuracy: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_model = model

    print(f"✅ Best Model ({model_type}) found with params: {best_params} | Best Accuracy: {best_score:.4f}")

    best_model.fit(X_train, y_train)
    print(f"✅ Model ({model_type}) trained with fine-tuning.")
    return best_model



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
    fine_tuned_model = train_model_with_fine_tuning(X_train, y_train, config_path='config.yaml')

    # Evaluate the trained model
    f1_score = evaluate_model(fine_tuned_model, X_test, y_test)

    print(f"\n🏆 Best model trained with an F1 Score of {f1_score:.2f}")
    save_model(fine_tuned_model, model_output_path)


    # fine_tuned_model = {
    #     model: train_model_with_fine_tuning(X_train, y_train, model, fine_tuning_params[model])
    #     for model in fine_tuning_params
    # }


if __name__ == "__main__":
    optimization_pipeline()


