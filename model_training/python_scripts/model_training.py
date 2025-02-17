# This Python script is designed to implement a machine learning pipeline that handles the following steps:
# 1) Data Splitting: Splits the dataset into training and testing sets.
# 2) Model Training: Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting) using the training data.
# 3) Evaluating Trained Models
# 4) Model Evaluation: Evaluates the models based on their accuracy, precision, recall, and F1 score.
# 5) Model Saving: Saves the best-performing model for later use.

import pandas as pd 
import joblib 
import os  
import sys  
import yaml 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Force immediate log output (disable buffering)
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# Load configuration from YAML file
# This allows flexibility by defining paths and parameters externally in a configuration file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file) # Load configuration as a dictionary

# Define paths and parameters from the configuration file
cleaned_train_data_path = config["paths"]["cleaned_train"]
model_output_path = config["paths"]["model_output"]
target_column = config["parameters"]["target_column"]
test_size = config["parameters"]["test_size"]
random_state = config["parameters"]["random_state"]
fine_tuning_params = config["fine_tuning"]

# ------------------------------------------------------------
# Function 1: Splitting the Data for Training and Testing
# ------------------------------------------------------------
# 
def split_data(data, target_column, test_size, random_state):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    print("✅ Data split into training and testing sets")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ------------------------------------------------------------
# Function 2: Training the Model
# ------------------------------------------------------------
def train_model(X_train, y_train, model_type):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    if model_type not in models:
        raise ValueError("❌ Invalid model_type.")

    model = models[model_type]
    model.fit(X_train, y_train)
    print(f"✅ Model ({model_type}) trained successfully.")
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
    directory = os.path.dirname(output_path)
    
    # Ensure `directory` is actually a directory, not a file
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    
    # Remove the existing file before saving a new model
    if os.path.exists(output_path):
        os.remove(output_path)

    joblib.dump(model, output_path)
    print(f"✅ Model saved to {output_path}")



# ------------------------------------------------------------
# Model Training Pipeline Execution
# ------------------------------------------------------------

def modeling_pipeline():
    if not os.path.exists(cleaned_train_data_path):
        raise FileNotFoundError(f"❌ File not found: {cleaned_train_data_path}")

    print("\n📂 Loading dataset...")
    cleaned_train_data = pd.read_csv(cleaned_train_data_path)
    print("✅ Dataset loaded!")

    X_train, X_test, y_train, y_test = split_data(cleaned_train_data, target_column, test_size, random_state)

    models = {
        "logistic_regression": train_model(X_train, y_train, "logistic_regression"),
        "random_forest": train_model(X_train, y_train, "random_forest"),
        "gradient_boosting": train_model(X_train, y_train, "gradient_boosting"),
    }

    for model_name, model in models.items():
        model_path = os.path.join(model_output_path, f"{model_name}.pkl")
        save_model(model, model_path)

if __name__ == "__main__":
    modeling_pipeline()
