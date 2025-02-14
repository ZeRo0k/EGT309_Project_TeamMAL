import streamlit as st
import joblib
import pandas as pd
import os
import yaml

# Get the current directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load model path from config
config_path = os.path.join(current_dir, 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

MODEL_PATH = os.path.join(current_dir, '..', 'saved_model', 'optimized_model.pkl')

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Error: Model file not found at {MODEL_PATH}")
    st.stop()

# Streamlit UI
st.title('🚢 Titanic Survival Prediction')
st.write('Input passenger details to get a survival prediction.')

# User Input Fields
pclass = st.selectbox('Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare Paid', min_value=0.0, value=0.0)
embarked = st.selectbox('Embarked Port', ['S', 'C', 'Q'])  # 🔥 Added missing input field

# Preprocess user input to match model's input features
def preprocess_input_data(user_input):
    """Ensure all required features are included and encoded properly."""
    
    # Convert user input dictionary into a DataFrame
    input_data = pd.DataFrame([user_input])  # ✅ FIXED: Convert dictionary to DataFrame

    # Compute Family Size
    input_data["Family Size"] = input_data["SibSp"] + input_data["Parch"]

    # Convert categorical features to numerical
    input_data["Sex"] = input_data["Sex"].map({"male": 1, "female": 0})
    input_data["Embarked"] = input_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Encode Title feature
    input_data["Title"] = 4  # Default to "Others" since names are not provided

    # Create Age Group (Ensure binning matches training)
    age_bins = [0, 5, 13, 20, 40, 60, 100]
    age_labels = [0, 1, 2, 3, 4, 5]
    input_data["Age Group"] = pd.cut(input_data["Age"], bins=age_bins, labels=age_labels).astype(float)

    # Create Fare Group (Ensure binning matches training)
    fare_bins = [-1, 10, 20, 50, 100, 1000]
    fare_labels = [0, 1, 2, 3, 4]
    input_data["Fare Group"] = pd.cut(input_data["Fare"], bins=fare_bins, labels=fare_labels).astype(float)

    # **Ensure feature order matches model's training order**
    required_features = list(model.feature_names_in_)
    input_data = input_data.reindex(columns=required_features, fill_value=0)  # **🔥 FIXED: Order features**

    return input_data

# Prediction Button
if st.button('🔍 Predict'):
    # Create dictionary from user input
    user_input = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }

    # Preprocess user input
    input_data = preprocess_input_data(user_input)

    # Debugging: Check input features vs model trained features
    print(f"Model trained with features: {model.feature_names_in_}")
    print(f"Input features provided (in correct order): {list(input_data.columns)}")

    # Predict survival
    prediction = model.predict(input_data)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    st.success(f'The model predicts: {result}')
