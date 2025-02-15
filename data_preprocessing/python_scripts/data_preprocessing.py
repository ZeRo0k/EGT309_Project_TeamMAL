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
test_data_path = config["paths"]["raw_test"]
cleaned_train_path = config["paths"]["cleaned_train"]
cleaned_test_path = config["paths"]["cleaned_test"]

# Load parameters
features_for_imputation = config["features"]["imputation"]
title_categories = config["features"]["title_categories"]
fare_bins = config["features"]["fare_bins"]
fare_labels = config["features"]["fare_labels"]
age_bins = config["features"]["age_bins"]
age_labels = config["features"]["age_labels"]
columns_to_drop = config["features"]["columns_to_drop"]

def clean_null(data):
    """Handle missing values with KNN Imputer & drop unwanted columns."""
    if 'Age' in data.columns:
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        data[features_for_imputation] = knn_imputer.fit_transform(data[features_for_imputation])

    if 'Cabin' in data.columns:
        data.drop("Cabin", axis=1, inplace=True)

    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].fillna('S')

    print("✅ Null values cleaned...", flush=True)
    return data

def data_transformation(data):
    """Convert categorical values into numeric representations."""
    if 'Sex' in data.columns:
        data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    print("✅ Data Transformation completed...", flush=True)
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
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"❌ Train or test data files not found in {train_data_path} or {test_data_path}")

    print("📂 Loading datasets...", flush=True)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    print("✅ Datasets loaded successfully!", flush=True)

    # Apply preprocessing steps in sequence
    print("\n🔄 Preprocessing train dataset...", flush=True)
    train_data = clean_null(train_data)
    train_data = data_transformation(train_data)
    train_data = feature_engineering(train_data)
    train_data = drop_columns(train_data)

    print("\n🔄 Preprocessing test dataset...", flush=True)
    test_data = clean_null(test_data)
    test_data = data_transformation(test_data)
    test_data = feature_engineering(test_data)

    # Ensure directories exist before saving
    os.makedirs(os.path.dirname(cleaned_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(cleaned_test_path), exist_ok=True)

    # Save cleaned datasets
    train_data.to_csv(cleaned_train_path, index=False)
    test_data.to_csv(cleaned_test_path, index=False)
    print(f"✅ Cleaned datasets saved:\n- {cleaned_train_path}\n- {cleaned_test_path}", flush=True)

    # Output summary
    print("\n📊 Preprocessed Data Summary:", flush=True)
    print("\nTrain Dataset Overview:", flush=True)
    print(train_data.head(), flush=True)
    print("\nTrain Dataset Info:", flush=True)
    print(train_data.info(), flush=True)

    print("\nTest Dataset Overview:", flush=True)
    print(test_data.head(), flush=True)
    print("\nTest Dataset Info:", flush=True)
    print(test_data.info(), flush=True)

    print("\n✅ Data Preprocessing Completed!", flush=True)
    exit(0)

if __name__ == "__main__":
    preprocessing_pipeline()




















