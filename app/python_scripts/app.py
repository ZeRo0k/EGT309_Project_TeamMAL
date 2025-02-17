import streamlit as st
import joblib
import pandas as pd
import os
import yaml
from sklearn.impute import KNNImputer

# Import preprocessing functions from data_preprocessing.py
from data_preprocessing.python_scripts.data_preprocessing import clean_null, data_transformation, feature_engineering, drop_columns

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

# Streamlit Sidebar for Navigation
st.sidebar.title("MAL Team")
app_mode = st.sidebar.radio("Select your preferrred mode", ["Single Prediction", "Batch Prediction"])

# UI (Single Prediction Page)
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

    # Preprocess user input to match model"s input features
    def preprocess_input_data(user_input):
        
        # Convert user input dictionary into a DataFrame
        input_data = pd.DataFrame([user_input])  # ✅ FIXED: Convert dictionary to DataFrame

        # Compute Family Size
        input_data["Family Size"] = input_data["SibSp"] + input_data["Parch"]

        # Encode Sex feature
        input_data["Sex"] = input_data["Sex"].map({"male": 1, "female": 0})

        # Encode Embarked feature
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

        # Ensure feature order matches model"s training order
        required_features = list(model.feature_names_in_)
        input_data = input_data.reindex(columns=required_features, fill_value=0)  # **🔥 FIXED: Order features**

        return input_data

    # Prediction Button
    if st.button("🔍 Predict"):
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
        st.success(f"The model predicts: {result}")

# UI for Batch Prediction
elif app_mode == "Batch Prediction":
    st.title("🚢 Titanic Survival Prediction (Batch)")
    st.write("Upload a CSV file with passenger data to get their survival prediction.")

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        file_df = pd.read_csv(uploaded_file)
        
        # Preprocess the batch data using the functions from preprocess.py
        def preprocess_batch_data(file_df):

            # Separate the columns into numeric and categorical
            numeric_columns = file_df.select_dtypes(include=["number"]).columns
            categorical_columns = file_df.select_dtypes(exclude=["number"]).columns

            # Apply preprocessing steps to the batch data
            file_df = clean_null(file_df)
            file_df = data_transformation(file_df)
            file_df = feature_engineering(file_df)
            file_df = drop_columns(file_df)

            # Ensure feature order matches model's training order
            required_columns = list(model.feature_names_in_)
            file_df = file_df.reindex(columns=required_columns, fill_value=0) 

            return file_df

        # Preprocess the data
        preprocessed_data = preprocess_batch_data(file_df)
        
        # Make predictions for all rows in the batch
        predictions = model.predict(preprocessed_data)

        # Add predictions to the DataFrame
        file_df.loc[:len(predictions)-1, "Survival Prediction"] = predictions
        st.write("Prediction Results:", file_df)

        # Allow users to download the results
        st.download_button(
            label="Download Results",
            data=file_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
