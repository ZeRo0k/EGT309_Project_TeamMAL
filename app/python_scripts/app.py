# This script builds a Streamlit web application for predicting Titanic survival outcomes using a trained machine learning model. 
# 
# Key Features:
# - Single Prediction Mode: Users can enter passenger details manually.
# - Batch Prediction Mode: Users can upload a CSV file for predictions.
# - Data Preprocessing: Uses the same preprocessing pipeline applied to the training dataset to ensure consistency.
# - Model Loading: Loads the trained and optimized machine learning model.
# - User-Friendly Interface: Designed for easy interaction using Streamlit.

import streamlit as st
import joblib
import pandas as pd
import os
import yaml

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_preprocessing', 'python_scripts'))
from data_preprocessing import handle_null, data_encoding, feature_engineering, drop_columns

# Load config
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

MODEL_PATH = config["paths"]["saved_model"]

# Debugging: Print the resolved model path
print(f"🔍 Checking model path: {MODEL_PATH}")

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Error: Model file not found at {MODEL_PATH}")

print("✅ Model found! Loading...")

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Error: Model file not found at {MODEL_PATH}")
    st.stop()


# -----------------------------------------------------------------------------
# Streamlit Sidebar Navigation
# -----------------------------------------------------------------------------
# Provides users with a selection menu to choose between:
# - Single Prediction: Enter details manually for one passenger.
# - Batch Prediction: Upload a CSV file for batch processing.
# -----------------------------------------------------------------------------
st.sidebar.title("MAL Team")
app_mode = st.sidebar.radio("Select your preferrred mode", ["Single Prediction", "Batch Prediction"])



# -----------------------------------------------------------------------------
# Single Passenger Prediction UI
# -----------------------------------------------------------------------------
if app_mode == "Single Prediction":
    st.title("🚢 Titanic Survival Prediction (Single)")
    st.write("Input passenger details to get a survival prediction.")

    # User Input Fields
    pclass = st.selectbox("Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare Paid", min_value=0.0, value=0.0)
    embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])  # 🔥 Added missing input field


    # -----------------------------------------------------------------------------
    # Function: preprocess_input_data
    # -----------------------------------------------------------------------------
    # - Converts user input into a Pandas DataFrame.
    # - Computes additional features such as Family Size.
    # - Encodes categorical variables (Sex, Embarked).
    # - Bins Age and Fare into categorized groups.
    # - Ensures feature order matches the trained model's requirements.
    # -----------------------------------------------------------------------------
    def preprocess_input_data(user_input):
        
        input_data = pd.DataFrame([user_input])  
        input_data["Family Size"] = input_data["SibSp"] + input_data["Parch"]
        input_data["Sex"] = input_data["Sex"].map({"male": 1, "female": 0})
        input_data["Embarked"] = input_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
        input_data["Title"] = 4  # Default to "Others" 
        
        age_bins = [0, 5, 13, 20, 40, 60, 100]
        age_labels = [0, 1, 2, 3, 4, 5]
        input_data["Age Group"] = pd.cut(input_data["Age"], bins=age_bins, labels=age_labels).astype(float)
        fare_bins = [-1, 10, 20, 50, 100, 1000]
        fare_labels = [0, 1, 2, 3, 4]
        input_data["Fare Group"] = pd.cut(input_data["Fare"], bins=fare_bins, labels=fare_labels).astype(float)

        required_features = list(model.feature_names_in_)
        input_data = input_data.reindex(columns=required_features, fill_value=0)  # **🔥 FIXED: Order features**

        return input_data

    # Prediction Button
    if st.button("🔍 Predict"):
        user_input = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked
        }

        # Preprocess and predict
        input_data = preprocess_input_data(user_input)
        print(f"Model trained with features: {model.feature_names_in_}")
        print(f"Input features provided (in correct order): {list(input_data.columns)}")

        prediction = model.predict(input_data)
        result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
        st.success(f"The model predicts: {result}")


# -----------------------------------------------------------------------------
# Batch Prediction UI
# -----------------------------------------------------------------------------
elif app_mode == "Batch Prediction":
    st.title("🚢 Titanic Survival Prediction (Batch)")
    st.write("Upload a CSV file with passenger data to get their survival prediction.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        file_df = pd.read_csv(uploaded_file)
        
        # -----------------------------------------------------------------------------
        # Function: preprocess_batch_data
        # -----------------------------------------------------------------------------
        # - Reads the uploaded CSV file.
        # - Applies preprocessing functions from `data_preprocessing.py`:
        #   - Handles missing values.
        #   - Encodes categorical variables.
        #   - Creates new features.
        #   - Drops unnecessary columns.
        # - Ensures feature order matches model training.
        # -----------------------------------------------------------------------------
        def preprocess_batch_data(file_df):

            file_df = handle_null(file_df)
            file_df = data_encoding(file_df)
            file_df = feature_engineering(file_df)
            file_df = drop_columns(file_df)

            required_columns = list(model.feature_names_in_)
            file_df = file_df.reindex(columns=required_columns, fill_value=0) 

            return file_df

        # Preprocess the data
        preprocessed_data = preprocess_batch_data(file_df)
        
        # Make predictions for all rows in the batch
        predictions = model.predict(preprocessed_data)
        file_df.loc[:len(predictions)-1, "Survival Prediction"] = predictions
        st.write("Prediction Results:", file_df)

        # Allow users to download the results
        st.download_button(
            label="Download Results",
            data=file_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
