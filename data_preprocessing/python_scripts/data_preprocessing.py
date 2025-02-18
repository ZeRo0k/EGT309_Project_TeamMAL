import pandas as pd
from sklearn.impute import KNNImputer
import re
import os
import sys
import yaml

# ✅ Ensure logs output immediately
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# Load YAML configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# File paths from config
train_data_path = config["paths"]["raw_train"]
cleaned_train_path = config["paths"]["cleaned_train"]

# Load parameters
features_for_imputation = config["features"]["imputation"]
title_categories = config["features"]["title_categories"]
fare_bins = config["features"]["fare_bins"]
fare_labels = config["features"]["fare_labels"]
age_bins = config["features"]["age_bins"]
age_labels = config["features"]["age_labels"]
columns_to_drop = config["features"]["columns_to_drop"]

def handle_null(data):
    """Handle missing values with KNN Imputer & drop unwanted columns."""
    if 'Fare' in data.columns and data['Fare'].isnull().sum() > 0:
        data['Fare'].fillna(data['Fare'].median(), inplace=True)

    """Handle missing values with KNN Imputer & drop unwanted columns."""
    if 'Age' in data.columns:
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        data[features_for_imputation] = knn_imputer.fit_transform(data[features_for_imputation])

    if 'Cabin' in data.columns:
        data.drop("Cabin", axis=1, inplace=True)

    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].fillna('S')

    print("✅ Null values handled...", flush=True)
    return data

def data_transformation(data):
    """Handle outliers for Age and Fare using the IQR method and remove extreme Fare values (>500)."""
    numerical_columns = ['Age', 'Fare']
    
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Clip outliers to the IQR range
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    
    # Remove extreme outliers for Fare
    data = data[data['Fare'] <= 500]
    
    print("✅ Data Transformation completed...", flush=True)
    return data

def data_encoding(data):
    """Convert categorical values into numeric representations."""
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    print("✅ Data Encoding completed...", flush=True)
    return data

def feature_engineering(data):
    """Create new features: Title, Family Size, Age Group, and Fare Group."""
    data['Title'] = data['Name'].apply(lambda x: re.search(r'([A-Za-z]+)\.', x).group(1) if re.search(r'([A-Za-z]+)\.', x) else "Unknown")
    data['Title'] = data['Title'].apply(lambda x: x if x in title_categories else 'Others')

    title_encoding = {title: idx for idx, title in enumerate(title_categories)}
    title_encoding["Others"] = len(title_categories)
    data['Title'] = data['Title'].map(title_encoding)

    data['Family Size'] = data['SibSp'] + data['Parch']
    data['Fare Group'] = pd.cut(data['Fare'], bins=fare_bins, labels=fare_labels)
    data['Age Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

    print("✅ Feature Engineering completed...", flush=True)
    return data

def drop_columns(data):
    """Drop irrelevant columns."""
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1)
    print("✅ Unnecessary columns dropped...", flush=True)
    return data

def preprocessing_pipeline():
    """Execute the complete preprocessing pipeline."""
    # Ensure data paths exist
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"❌ Train data file not found in {train_data_path}")

    print("📂 Loading datasets...", flush=True)
    train_data = pd.read_csv(train_data_path)
    print("✅ Datasets loaded successfully!", flush=True)

    # Apply preprocessing steps in sequence
    print("\n🔄 Preprocessing train dataset...", flush=True)
    train_data = handle_null(train_data)
    train_data = data_transformation(train_data)
    train_data = data_encoding(train_data)
    train_data = feature_engineering(train_data)
    train_data = drop_columns(train_data)

    # Ensure directories exist before saving
    os.makedirs(os.path.dirname(cleaned_train_path), exist_ok=True)

    # Save cleaned datasets
    train_data.to_csv(cleaned_train_path, index=False)
    print(f"✅ Cleaned datasets saved:\n- {cleaned_train_path}", flush=True)

    # Output summary
    print("\n📊 Preprocessed Data Summary:", flush=True)
    print("\nTrain Dataset Overview:", flush=True)
    print(train_data.head(), flush=True)
    print("\nTrain Dataset Info:", flush=True)
    print(train_data.info(), flush=True)

    print("\n✅ Data Preprocessing Completed!", flush=True)
    exit(0)

if __name__ == "__main__":
    preprocessing_pipeline()