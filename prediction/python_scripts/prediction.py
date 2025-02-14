import pickle
import pandas as pd
import yaml
import os

# Determine execution environment (Kubernetes vs Local)
RUNNING_IN_K8S = os.getenv("MODEL_PATH") is not None

if not RUNNING_IN_K8S:
    # Local Execution - Load YAML config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "prediction_config.yaml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    MODEL_PATH = os.path.join(current_dir, config["paths"]["saved_model"])
else:
    # Kubernetes Execution - Load from Environment Variables
    MODEL_PATH = os.getenv("MODEL_PATH")

# Load Model
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    exit(1)

# Get User Input
def get_user_input():
    print("\n💡 Enter Passenger Details for Survival Prediction:\n")
    pclass = int(input("Pclass (1/2/3): "))
    sex = input("Sex (male/female): ").strip().lower()
    age = float(input("Age: "))
    sibsp = int(input("Siblings/Spouses aboard: "))
    parch = int(input("Parents/Children aboard: "))
    fare = float(input("Fare Paid: "))

    # Convert categorical to numerical
    sex = 1 if sex == "male" else 0

    # Create DataFrame
    data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], 
                        columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    return data

# Make Prediction
def make_prediction(data):
    prediction = model.predict(data)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    return result

# CLI Execution
def cli_mode():
    """Runs CLI-based prediction."""
    data = get_user_input()
    result = make_prediction(data)
    print("\n🛳 Prediction:", result)

# Batch File Prediction Mode
def batch_mode(file_path):
    """Runs batch prediction on a CSV file."""
    try:
        data = pd.read_csv(file_path)
        predictions = model.predict(data)
        data["Prediction"] = ["✅ Survived" if pred == 1 else "❌ Did Not Survive" for pred in predictions]
        
        output_path = file_path.replace(".csv", "_predictions.csv")
        data.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error processing batch file: {e}")

if __name__ == "__main__":
    mode = input("\n🔍 Select mode (1: Interactive CLI, 2: Batch CSV Prediction): ").strip()
    
    if mode == "1":
        cli_mode()
    elif mode == "2":
        file_path = input("📂 Enter CSV file path: ").strip()
        batch_mode(file_path)
    else:
        print("❌ Invalid choice! Please select 1 or 2.")
