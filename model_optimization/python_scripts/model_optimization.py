# This Python script handles the training of the optimized machine learning models
# 1) Data Splitting: Splits the dataset into training and testing sets.
# 2) Model Optimization: Train models with hyper parameter tuning and perform voting ensembler 
# 3) Model Evaluation: Evaluates the models based on their accuracy, precision, recall, and F1 score.
# 4) Save Model: Saves best mode into selected directory

# Import libraries
import pandas as pd
import numpy as np
import yaml
import os
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedShuffleSplit


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

# Split the data into training and testing sets with optional stratified sampling
# Train size: 80% of the dataset.
# Test size: 20% of the dataset.

def split_data(data, target_column, test_size, random_state):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    print("✅ Data split into training and testing sets")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ------------------------------------------------------------
# Function 2: Training the Model with Fine-Tuning
# ------------------------------------------------------------

# Model Fine-Tuning Process:
    # 1. Extract hyperparameters from config for each model type.
    # 2. Perform Stratified Shuffle Split cross-validation.
    # 3. Define the base models: Logistic Regression, Random Forest, Gradient Boosting.
    # 4. Iterate over hyperparameter combinations for each model.
    # 5. Train each model using cross-validation.
    # 6. Evaluate performance using F1 Score and select the best model from each fold.
    # 7. Retrain the best model on the full dataset.
    # 8. Store the best models for each algorithm.
    # 9. Create a weighted voting ensemble using the best models.
    # 10. Train the voting classifier with the final best models.
    # 11. Evaluate and print final model metrics for each model.

# Training the model based on specified type:
    # Logistic Regression: Simple and interpretable, used as a baseline model.
    # C: [0.01, 0.1, 0.5, 1.0]  # Regularization strength (smaller = stronger).
    # penalty: ["elasticnet"]  # Mix of L1 and L2 regularization.
    # solver: ["saga"]  # Works best with elasticnet.
    # l1_ratio: [0.1, 0.5, 0.7]  # Balance between L1 (sparse) and L2 (smooth).
    # max_iter: [5000]  # Max iterations to ensure convergence.

    # Random Forest: Ensemble of decision trees for better stability.
    # n_estimators: [50, 100, 200]  # Number of trees (more = better but slower).
    # max_depth: [5, 10, 15]  # Limits tree depth to avoid overfitting.
    # min_samples_split: [2, 5]  # Min samples needed to split a node.

    # Gradient Boosting: Builds trees sequentially to minimize errors.
    # n_estimators: [50, 100, 200]  # More trees improve accuracy but take longer.
    # max_depth: [3, 4, 5]  # Keeps trees shallow to prevent overfitting.
    # min_samples_split: [2, 5]  # Fewer samples allow deeper splits.
    # learning_rate: [0.01, 0.05, 0.1]  # Step size per iteration (lower = stable).

    # Cross-Validation: Splits data multiple times for better evaluation.
    # n_splits: 7  # More folds = more stable results.
    # shuffle: true  # Randomizes data before splitting.
    # random_state: 42  # Ensures reproducibility.

# Create voting ensembler 

def train_model_with_fine_tuning(X_train, y_train, config):
    # Extract hyperparameter configurations
    logistic_regression_params = config["logistic_regression"]
    random_forest_params = config["random_forest"]
    gradient_boosting_params = config["gradient_boosting"]
    cv_params = config["cross_validation"]

    # Set up Stratified Shuffle Split for cross-validation
    kfold = StratifiedShuffleSplit(n_splits=cv_params['n_splits'], test_size=test_size, random_state=cv_params['random_state'])

    # Define base models and their hyperparameters
    base_models = {
        "logistic_regression": (LogisticRegression, logistic_regression_params),
        "random_forest": (RandomForestClassifier, random_forest_params),
        "gradient_boosting": (GradientBoostingClassifier, gradient_boosting_params)
    }

    best_models = {} # Store best models 

    for model_name, (model_class, param_grid) in base_models.items():
        param_combinations = list(ParameterGrid(param_grid))

        best_score = 0
        best_model = None

        for params in param_combinations:
            scores = []
            fold_models = []

            # Perform cross-validation
            for train_index, test_index in kfold.split(X_train, y_train):
                X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
                y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]

                model = model_class(**params)
                model.fit(X_fold_train, y_fold_train)
                score = f1_score(y_fold_test, model.predict(X_fold_test)) 
                scores.append(score)
                fold_models.append(model)

            # Select the best-performing model across folds
            best_fold_index = np.argmax(scores)
            if scores[best_fold_index] > best_score:
                best_score = scores[best_fold_index]
                best_model = fold_models[best_fold_index]

        # Retrain the best model using the full training set
        best_model.fit(X_train, y_train)
        best_models[model_name] = best_model
        print(f"✅ Best {model_name}: F1 Score = {best_score:.4f}")

    # Create a weighted voting classifier using the best models
    voting_ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='hard',
        weights=[0.2, 1.5, 1.5],
        n_jobs=-1
    )

    # Train the ensemble model
    voting_ensemble.fit(X_train, y_train)
    print("✅ Weighted voting ensemble trained with best models.")

    # Evaluate the best models individually
    print("\n🔍 Final Evaluation Metrics:")
    for model_name, model in best_models.items():
        print(f"\n{model_name}:")
        evaluate_model(model, X_train, y_train)

    return voting_ensemble

# ------------------------------------------------------------
# Function 3: Evaluating trained models
# ------------------------------------------------------------

# Evaluate the model based on specified evaluation metrics:
# - Accuracy: Overall correctness of predictions.
# - Precision: Accuracy of positive predictions.
# - Recall: Captures all actual positives.
# - F1 Score: Balances precision and recall.

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

# Saving best model to specified file path

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

    # Evaluate the voting ensemble
    print("\nVoting Ensemble:")
    f1_score = evaluate_model(fine_tuned_model, X_test, y_test)

    print(f"\n🏆 Best model trained with an F1 Score of {f1_score:.2f}")
    save_model(fine_tuned_model, model_output_path)

if __name__ == "__main__":
    optimization_pipeline()