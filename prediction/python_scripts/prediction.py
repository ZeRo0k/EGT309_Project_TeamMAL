import pickle
import pandas as pd
import yaml
import os
import joblib

# ✅ Load configuration from YAML file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# ✅ Determine execution environment (Kubernetes vs Local)
RUNNING_IN_K8S = os.getenv("MODEL_PATH") is not None

# ✅ Set model file path
MODEL_PATH = os.getenv("MODEL_PATH") if RUNNING_IN_K8S else os.path.join(current_dir, config["paths"]["saved_model"])

# ✅ Debugging Output
print(f"🔍 Looking for model at: {MODEL_PATH}")

# ✅ Load the model
try:
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)  # Use joblib instead of pickle for sklearn models
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    exit(1)

# ✅ Get User Input (Interactive Mode)
def get_user_input():
    print("\n💡 Enter Passenger Details for Survival Prediction:\n")
    pclass = int(input("Pclass (1/2/3): "))

    # Ensure valid input for sex
    while True:
        sex = input("Sex (male/female): ").strip().lower()
        if sex in ["male", "female"]:
            break
        print("❌ Invalid input. Please enter 'male' or 'female'.")

    age = float(input("Age: "))
    sibsp = int(input("Siblings/Spouses aboard: "))
    parch = int(input("Parents/Children aboard: "))
    fare = float(input("Fare Paid: "))

    # Ensure valid input for embarked
    while True:
        embarked = input("Embarked (S/C/Q): ").strip().upper()
        if embarked in ["S", "C", "Q"]:
            embarked = {"S": 0, "C": 1, "Q": 2}[embarked]
            break
        print("❌ Invalid input. Please enter 'S', 'C', or 'Q'.")

    # Convert categorical variables to numerical
    sex = 1 if sex == "male" else 0

    # Extract Title (since Name is missing, assume based on Sex)
    title = 0 if sex == 1 else 2  # Assume "Mr" for male (0), "Miss" for female (2)

    # Compute Family Size
    family_size = sibsp + parch

    # Apply binning for Age Group
    age_bins = [0, 5, 13, 20, 40, 60, 100]
    age_labels = [0, 1, 2, 3, 4, 5]
    age_group = pd.cut([age], bins=age_bins, labels=age_labels).astype(float)[0]

    # Apply binning for Fare Group
    fare_bins = [-1, 10, 20, 50, 100, 1000]
    fare_labels = [0, 1, 2, 3, 4]
    fare_group = pd.cut([fare], bins=fare_bins, labels=fare_labels).astype(float)[0]

    # Create DataFrame with expected features
    data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked, title, family_size, fare_group, age_group]], 
                        columns=config["features"]["input_features"])

    return data



# ✅ Make Prediction
def make_prediction(data):
    prediction = model.predict(data)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    return result

# ✅ CLI Execution
def cli_mode():
    """Runs CLI-based prediction."""
    data = get_user_input()
    result = make_prediction(data)
    print("\n🛳 Prediction:", result)

# ✅ Batch File Prediction Mode
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

# ✅ Main Execution Logic
if __name__ == "__main__":
    mode = input("\n🔍 Select mode (1: Interactive CLI, 2: Batch CSV Prediction): ").strip()
    
    if mode == "1":
        cli_mode()
    elif mode == "2":
        file_path = input("📂 Enter CSV file path: ").strip()
        batch_mode(file_path)
    else:
        print("❌ Invalid choice! Please select 1 or 2.")
