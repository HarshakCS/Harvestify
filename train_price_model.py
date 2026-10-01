import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import argparse

def load_data(filepath):
    """Load and preprocess the dataset"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Convert date to datetime and extract features
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])
    df['Year'] = df['Arrival_Date'].dt.year
    df['Month'] = df['Arrival_Date'].dt.month
    df['Day'] = df['Arrival_Date'].dt.day
    df['DayOfWeek'] = df['Arrival_Date'].dt.dayofweek
    df['DayOfYear'] = df['Arrival_Date'].dt.dayofyear
    
    return df

def preprocess_data(df):
    """Preprocess the data for modeling"""
    print("Preprocessing data...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.dropna(subset=['Modal_Price'])
    
    # Encode categorical variables
    categorical_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Select features and target
    features = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 
                'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']
    target = 'Modal_Price'
    
    X = df_processed[features]
    y = df_processed[target]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, X_test, y_train, y_test, label_encoders, scaler

def train_model(X_train, y_train):
    """Train a Random Forest model"""
    print("Training model...")
    
    # Initialize the model with optimized hyperparameters
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance"""
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def save_model(model, label_encoders, scaler, output_dir='models'):
    """Save the trained model and preprocessing objects"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'crop_price_predictor.pkl')
    joblib.dump(model, model_path)
    
    # Save label encoders
    le_path = os.path.join(output_dir, 'label_encoders.pkl')
    joblib.dump(label_encoders, le_path)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel and preprocessing objects saved to {output_dir}/")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train a crop price prediction model')
    parser.add_argument('--data', type=str, default='Data-processed/consolidated_commodity_prices.csv',
                       help='Path to the input CSV file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save the trained model and preprocessing objects')
    args = parser.parse_args()
    
    try:
        # Load and preprocess data
        df = load_data(args.data)
        X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_data(df)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model and preprocessing objects
        save_model(model, label_encoders, scaler, args.output_dir)
        
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
