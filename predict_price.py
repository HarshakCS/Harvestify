import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

class CropPricePredictor:
    def __init__(self, model_dir='models'):
        """Initialize the predictor with trained model and preprocessing objects"""
        self.model = None
        self.label_encoders = None
        self.scaler = None
        self.is_initialized = False
        self.model_dir = model_dir
        
        try:
            # Check if model directory exists
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found: {os.path.abspath(model_dir)}")
                
            # Check for required files
            required_files = ['crop_price_predictor.pkl', 'label_encoders.pkl', 'scaler.pkl']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
            
            if missing_files:
                raise FileNotFoundError(
                    f"Missing required model files: {', '.join(missing_files)}\n"
                    f"Model directory: {os.path.abspath(model_dir)}\n"
                    f"Files in directory: {os.listdir(model_dir) if os.path.exists(model_dir) else 'Directory does not exist'}"
                )
            
            # Load model and preprocessing objects
            print(f"Loading model from: {os.path.abspath(model_dir)}")
            self.model = joblib.load(os.path.join(model_dir, 'crop_price_predictor.pkl'))
            self.label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            
            # Verify all required label encoders are present
            required_encoders = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
            missing_encoders = [e for e in required_encoders if e not in self.label_encoders]
            if missing_encoders:
                raise ValueError(f"Missing label encoders for: {', '.join(missing_encoders)}")
                
            self.is_initialized = True
            print("Model and preprocessing objects loaded successfully!")
            
        except Exception as e:
            error_msg = f"Error initializing CropPricePredictor: {str(e)}"
            print(error_msg)
            # Instead of raising, we'll set is_initialized to False
            self.is_initialized = False
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        try:
            # Convert to DataFrame if it's a dictionary
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()
            
            # Ensure all required columns are present
            required_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Arrival_Date']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Convert date to datetime and extract features
            try:
                df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], errors='coerce')
                if df['Arrival_Date'].isnull().any():
                    raise ValueError("Invalid date format in Arrival_Date")
                    
                df['Year'] = df['Arrival_Date'].dt.year
                df['Month'] = df['Arrival_Date'].dt.month
                df['Day'] = df['Arrival_Date'].dt.day
                df['DayOfWeek'] = df['Arrival_Date'].dt.dayofweek
                df['DayOfYear'] = df['Arrival_Date'].dt.dayofyear
            except Exception as e:
                raise ValueError(f"Error processing date: {str(e)}")
            
            # Clean and validate categorical variables
            categorical_cols = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
            for col in categorical_cols:
                if col in df.columns:
                    # Convert to string and strip whitespace
                    df[col] = df[col].astype(str).str.strip()
                    # Replace empty strings with the most frequent category
                    most_frequent = self.label_encoders[col].classes_[0]  # Use first class as fallback
                    df[col] = df[col].replace('', most_frequent)
                    # Handle unseen labels by using the most frequent category
                    df[col] = df[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else most_frequent
                    )
                    try:
                        df[col] = self.label_encoders[col].transform(df[col])
                    except ValueError as e:
                        # If still getting unknown labels, use the most frequent category
                        print(f"Warning: Found unknown labels in {col}, using most frequent category")
                        df[col] = self.label_encoders[col].transform([most_frequent] * len(df))
            
            # Select features in the correct order
            features = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 
                      'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']
            
            # Ensure all required features are present
            for feature in features:
                if feature not in df.columns:
                    df[feature] = 0  # Default value for missing features
            
            # Scale numerical features
            numerical_cols = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']
            if len(numerical_cols) > 0 and not df[numerical_cols].empty:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
            return df[features]
            
        except Exception as e:
            error_msg = f"Error in preprocess_input: {str(e)}"
            print(error_msg)
            raise
    
    def predict(self, input_data):
        """Make price predictions"""
        if not self.is_initialized:
            raise RuntimeError("Predictor is not properly initialized. Please check the model files.")
        
        if not input_data:
            raise ValueError("Input data cannot be empty")
            
        try:
            print(f"Making prediction with input data: {input_data}")
            
            # Ensure all required fields are present and not empty
            required_fields = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Arrival_Date']
            for field in required_fields:
                if field not in input_data or not input_data[field]:
                    raise ValueError(f"Missing or empty required field: {field}")
            
            # Create a copy of input data to avoid modifying the original
            input_copy = {k: str(v).strip() for k, v in input_data.items()}
            
            # Preprocess input
            X = self.preprocess_input(input_copy)
            if X.empty:
                raise ValueError("Failed to preprocess input data")
                
            print(f"Preprocessed features: {X.to_dict(orient='records')[0] if not X.empty else 'No features'}")
            
            # Make prediction
            prediction = self.model.predict(X)
            
            # Inverse transform the prediction if needed
            if hasattr(self.scaler, 'inverse_transform'):
                try:
                    # Reshape prediction to 2D array with shape (n_samples, 1)
                    prediction_reshaped = prediction.reshape(-1, 1)
                    # Inverse transform
                    prediction = self.scaler.inverse_transform(prediction_reshaped).flatten()
                except ValueError as ve:
                    # If inverse transform fails with shape error, try with a different approach
                    print(f"Warning: Could not inverse transform with error: {ve}")
                    print("Trying alternative inverse transform method...")
                    try:
                        # Create a 2D array with the same number of features as the scaler expects
                        dummy_features = np.zeros((1, self.scaler.scale_.shape[0]))
                        dummy_features[0, 0] = prediction[0]  # Set the first feature to our prediction
                        prediction = self.scaler.inverse_transform(dummy_features)[:, 0]  # Get first column
                    except Exception as e:
                        print(f"Alternative inverse transform also failed: {e}")
                        # If all else fails, return the raw prediction
                        print("Returning raw prediction (may not be in the original scale)")
            
            print(f"\n=== Prediction: {prediction[0] if len(prediction) == 1 else prediction} ===")
            return float(prediction[0]) if len(prediction) == 1 else prediction.tolist()
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print("\n=== ERROR ===")
            print(error_msg)
            print("\n=== Input Data ===")
            for k, v in input_data.items():
                print(f"{k}: {v} ({type(v).__name__})")
            print("\n=== Traceback ===")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error during prediction: {error_msg}\n"
                            f"Input data that caused the error: {input_data}") from e

if __name__ == "__main__":
    import os
    
    # Get the absolute path to the models directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    predictor = CropPricePredictor(models_dir)
    
    # Example input (modify these values as needed)
    input_data = {
        'State': 'Karnataka',
        'District': 'Hassan',
        'Market': 'Arakalgud',
        'Commodity': 'Tomato',
        'Variety': 'Tomato',
        'Grade': 'Faq',
        'Arrival_Date': '2023-12-15'
    }
    
    try:
        # Make prediction
        predicted_price = predictor.predict(input_data)
        # Use 'INR' instead of the Rupee symbol to avoid encoding issues
        print(f"\nPredicted Price: INR {predicted_price:.2f} per quintal")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    example_usage()
