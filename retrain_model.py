import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def load_data():
    """Load the crop recommendation dataset"""
    # Try to find the dataset file in various locations
    possible_paths = [
        os.path.join('app', 'data', 'crop_recommendation.csv'),
        os.path.join('data', 'crop_recommendation.csv'),
        os.path.join('Data-processed', 'crop_recommendation.csv'),
        os.path.join('Data-raw', 'MergeFileCrop.csv'),
        os.path.join('Data-raw', 'cpdata.csv')
    ]
    
    for data_path in possible_paths:
        if os.path.exists(data_path):
            print(f"Found dataset at: {data_path}")
            df = pd.read_csv(data_path)
            # Check if the dataframe has the expected columns
            required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
            if all(col in df.columns for col in required_columns):
                return df
            else:
                print(f"Warning: {data_path} doesn't have all required columns")
    
    raise FileNotFoundError("Could not find a valid crop recommendation dataset")

def preprocess_data(df):
    """Preprocess the data for training"""
    # Convert labels to categorical codes
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def train_model(X_train, y_train):
    """Train a Random Forest Classifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    return accuracy

def save_model(model, feature_names, model_path):
    """Save the trained model and feature names"""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names to a separate file
    feature_path = os.path.join(os.path.dirname(model_path), 'feature_names.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"Model saved to {model_path}")
    print(f"Feature names saved to {feature_path}")

def main():
    try:
        print("Loading data...")
        df = load_data()
        
        print("\nPreprocessing data...")
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        
        print("\nTraining model...")
        model = train_model(X_train, y_train)
        
        print("\nEvaluating model...")
        accuracy = evaluate_model(model, X_test, y_test)
        
        # Save the model
        model_path = os.path.join('app', 'models', 'retrained_random_forest.pkl')
        save_model(model, feature_names, model_path)
        
        print("\nRetraining completed successfully!")
        
    except Exception as e:
        print(f"Error during model retraining: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
