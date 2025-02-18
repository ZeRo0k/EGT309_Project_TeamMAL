# This Python script is designed to implement a machine learning pipeline that handles the following steps:
# 1) Data Splitting: Splits the dataset into training and testing sets.
# 2) Model Training: Trains multiple models (Logistic Regression, Random Forest, Gradient Boosting) using the training data.
# 3) Evaluating Trained Models
# 4) Model Evaluation: Evaluates the models based on their accuracy, precision, recall, and F1 score.
# 5) Model Saving: Saves the trained models for later use.

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
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file) # Load configuration as a dictionary

# Extract paths and parameters from the configuration file
cleaned_train_data_path = config["paths"]["cleaned_train"]
model_output_path = config["paths"]["model_output"]
target_column = config["parameters"]["target_column"]
test_size = config["parameters"]["test_size"]
random_state = config["parameters"]["random_state"]
fine_tuning_params = config["fine_tuning"]

# ------------------------------------------------------------
# Function 1: Splitting the Data for Training and Testing
# ------------------------------------------------------------
# The dataset is divided into features (X) and target variable (y). The "train_test_split" function is used to randomly partition
# the data based on 80% for training and 20% for testing as defined in the "config.yaml". After that, the function returns four 
# variables which are "X_train", "X_test", "y_train", and "y_test". A print statement indicates the successful completion of the 
# data split. 

def split_data(data, target_column, test_size, random_state):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    print("✅ Data split into training and testing sets")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# ------------------------------------------------------------
# Function 2: Training the Model
# ------------------------------------------------------------
# This function trains a machine learning model based on the specified model_type. It takes in the training features (X_train), 
# the target labels (y_train), and the type of model to train. The function supports three models: Logistic Regression (a baseline 
# model for linear relationships), Random Forest (a robust ensemble method that captures non-linear patterns), and 
# Gradient Boosting (a more sophisticated, iterative approach for improved accuracy). It selects the model based on the input, trains
# it using the provided data, and returns the trained model. If an invalid model_type is provided, the function raises an error.

def train_model(X_train, y_train, model_type):

    # Define a dictionary of models to be trained, indexed by model type.
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    # Check if the provided model type is valid. If not, raise an error.
    if model_type not in models:
        raise ValueError("❌ Invalid model_type.")

    # Retrieve the corresponding model from the dictionary based on the provided model_type.
    model = models[model_type]

    # Train the selected model using the training data (X_train, y_train).
    model.fit(X_train, y_train)

    # Print a success message after the model is trained.
    print(f"✅ Model ({model_type}) trained successfully.")

    # Return the trained model
    return model

# ------------------------------------------------------------
# Function 3: Evaluating trained models
# ------------------------------------------------------------
# This function Assesses the model’s performance using key classification metrics: accuracy, precision, recall, and F1-score. 
# These metrics help in understanding how well the model generalizes to new data.

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
# Function 4: Saving the Trained Models
# ------------------------------------------------------------
# This function saves the trained model as a .pkl file to the specified output directory. If the directory doesn’t exist, 
# it is created, and any existing model file is replaced to ensure the latest version is stored.

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
# Executes the entire pipeline by:
# 1) Loading the dataset from a predefined path.
# 2) Splitting the data into training and testing sets.
# 3) Training multiple models (Logistic Regression, Random Forest, Gradient Boosting).
# 4) Saving the trained models for future use.

def modeling_pipeline():
    if not os.path.exists(cleaned_train_data_path):
        raise FileNotFoundError(f"❌ File not found: {cleaned_train_data_path}")

    print("\n📂 Loading dataset...")
    cleaned_train_data = pd.read_csv(cleaned_train_data_path)
    print("✅ Dataset loaded!\n")

    X_train, X_test, y_train, y_test = split_data(cleaned_train_data, target_column, test_size, random_state)

    models = {
        "logistic_regression": train_model(X_train, y_train, "logistic_regression"),
        "random_forest": train_model(X_train, y_train, "random_forest"),
        "gradient_boosting": train_model(X_train, y_train, "gradient_boosting"),
    }

    # Evaluate each model
    print("\n📊 Model Evaluation:")
    for model_name, model in models.items():
        print(f"\n🔍 Evaluating {model_name}...")
        evaluate_model(model, X_test, y_test)

    # Save the model
    for model_name, model in models.items():
        model_path = os.path.join(model_output_path, f"{model_name}.pkl")
        save_model(model, model_path)

if __name__ == "__main__":
    modeling_pipeline()