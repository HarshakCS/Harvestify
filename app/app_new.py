import os
import sys
import pickle
import numpy as np
import requests
from PIL import Image

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import local modules
try:
    from app.utils.disease import disease_dic
    from app.utils.fertilizer import fertilizer_dic
    from app.utils.plant_disease_keras import plant_disease_predictor, PLANTDNET_DISEASE_CLASSES
except ImportError:
    # Fallback to relative imports if absolute imports fail
    from utils.disease import disease_dic
    from utils.fertilizer import fertilizer_dic
    from utils.plant_disease_keras import plant_disease_predictor, PLANTDNET_DISEASE_CLASSES

# ==============================================================================================
# MODEL LOADING
# ==============================================================================================

def load_model():
    """Load the crop recommendation model"""
    try:
        print("\n=== Loading Crop Recommendation Model ===")
        print(f"Current directory: {os.getcwd()}")
        model_dir = os.path.join(project_root, 'app', 'models')
        print(f"Model directory: {model_dir}")
        
        # Check if models directory exists
        if not os.path.exists(model_dir):
            print("\nERROR: Models directory does not exist!")
            print(f"Looking for directory at: {model_dir}")
            print("Please ensure the 'models' directory exists in the app directory.")
            return None
        
        # List all files in the models directory
        print("\nContents of models directory:")
        model_files = []
        for f in os.listdir(model_dir):
            model_files.append(f)
            print(f"- {f}")
        
        # Try to load the retrained model first
        model_path = os.path.join(model_dir, 'retrained_random_forest.pkl')
        feature_path = os.path.join(model_dir, 'feature_names.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(feature_path):
            print("\nERROR: Retrained model or feature names file not found!")
            print(f"Looking for: {model_path} and {feature_path}")
            return None
        
        print(f"\nUsing retrained model: {os.path.basename(model_path)}")
        
        # Get file sizes
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
        feature_size = os.path.getsize(feature_path) / 1024  # Convert to KB
        print(f"Model file size: {model_size:.2f} MB")
        print(f"Feature names file size: {feature_size:.2f} KB")
        
        # Load the model
        print("\nLoading model and feature names...")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(feature_path, 'rb') as f:
                feature_names = pickle.load(f)
                
            print("Model and feature names loaded successfully!")
            print(f"Model type: {type(model)}")
            print(f"Feature names: {feature_names}")
            
            # Test the model with sample data
            if hasattr(model, 'predict'):
                sample_input = np.array([[50, 50, 50, 25, 70, 7, 150]])
                print(f"\nTesting with sample input: {sample_input}")
                sample_pred = model.predict(sample_input)
                print(f"Sample prediction: {sample_pred[0]}")
                
                if hasattr(model, 'predict_proba'):
                    sample_proba = model.predict_proba(sample_input)
                    print("Sample prediction probabilities:")
                    for i, prob in enumerate(sample_proba[0]):
                        print(f"  {model.classes_[i]}: {prob:.4f}")
            
            return model
            
        except Exception as e:
            print(f"\n=== ERROR LOADING MODEL ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nStack trace:")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"\n=== ERROR LOADING MODEL ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        print("==========================")
        return None

# Load the models when this module is loaded
crop_recommendation_model = load_model()

# Disease classes (using PlantDNet classes)
disease_classes = PLANTDNET_DISEASE_CLASSES

# Initialize PlantDNet disease prediction model
print("\n=== Loading PlantDNet Disease Prediction Model ===")
disease_model = plant_disease_predictor
if True:
    print("PlantDNet disease prediction model loaded successfully!")
else:
    print("Warning: PlantDNet disease prediction model could not be loaded")
    disease_model = None

# ==============================================================================================
# HELPER FUNCTIONS
# ==============================================================================================

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name: Name of the city to fetch weather for
    :return: tuple of (temperature, humidity) or default values (25.0, 70.0) if failed
    """
    # Default values if API call fails or no API key is found
    default_temp = 25.0
    default_humidity = 70.0
    
    try:
        api_key = os.environ.get('WEATHER_API_KEY')
        if not api_key:
            print("Warning: WEATHER_API_KEY not found in environment variables. Using default weather values.")
            return default_temp, default_humidity
            
        base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}"
        response = requests.get(base_url, timeout=5)  # Add timeout to prevent hanging
        data = response.json()
        
        if data.get("cod") == 200:
            temp = data.get("main", {}).get("temp", default_temp + 273.15) - 273.15  # Convert from Kelvin to Celsius
            humidity = data.get("main", {}).get("humidity", default_humidity)
            return temp, humidity
        else:
            print(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
            return default_temp, default_humidity
    except requests.exceptions.RequestException as e:
        print(f"Network error in weather_fetch: {str(e)}")
    except (KeyError, ValueError, AttributeError) as e:
        print(f"Error parsing weather data: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in weather_fetch: {str(e)}")
        
    return default_temp, default_humidity

def predict_image(img_path, model=None):
    """
    Predicts disease label using PlantDNet Keras model
    :params: img_path: Path to the image file
             model: Optional model to use for prediction (uses disease_model if None)
    :return: tuple of (prediction, confidence) where prediction is a string and confidence is a float
    """
    if model is None:
        model = disease_model
        
    if model is None:
        print("Error: Disease prediction model is not available")
        return "Disease prediction model not available", 0.0
        
    try:
        # Check if image exists
        if not os.path.exists(img_path):
            print(f"Error: Image file not found at {img_path}")
            return f"Error: Image file not found at {img_path}", 0.0
        
        print(f"\n=== Loading Image ===")
        print(f"Image path: {os.path.abspath(img_path)}")
        
        # Use the PlantDNet predictor to make prediction
        result = model.predict_disease(img_path)
        
        if result['success']:
            predicted_class = result['disease']
            confidence = result['confidence']
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            return predicted_class, confidence
        else:
            error_msg = f"Prediction failed: {result['error']}"
            print(error_msg)
            return error_msg, 0.0
            
    except Exception as e:
        error_msg = f"Unexpected error in predict_image: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, 0.0
