# Importing essential libraries and modules

from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.plant_disease_keras import plant_disease_predictor
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

# Disease classes matching the PyTorch model (38 classes)
disease_classes = [
    'Apple___Apple_scab',
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# import os
# import sys
# import traceback
import os
import sys
import traceback
from huggingface_hub import hf_hub_download

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
disease_model_path = hf_hub_download(
    repo_id="HARSHAKCS/harvestify-models",
    filename="plant_disease_model.pth"
)
# disease_model_path = os.path.join(current_dir, 'models', 'plant_disease_model.pth')

try:
    print(f"Loading model from: {disease_model_path}")
    
    # Check if model file exists
    if not os.path.exists(disease_model_path):
        raise FileNotFoundError(f"Model file not found at {disease_model_path}")
    
    # Initialize model with correct number of classes (38 as shown in the saved model)
    disease_model = ResNet9(3, 38)  # The model was trained with 38 classes
    
    # Load model weights with compatibility handling
    try:
        # Try loading with different approaches for compatibility
        state_dict = torch.load(disease_model_path, map_location=torch.device('cpu'))
        
        # Handle different save formats
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
        
        # Try to load state dict directly first
        try:
            disease_model.load_state_dict(state_dict)
        except Exception as load_error:
            print(f"Direct state dict loading failed: {str(load_error)}")
            print("Trying to fix key names...")
            
            # Fix potential key name issues (remove 'module.' prefix if present)
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    fixed_state_dict[key[7:]] = value  # Remove 'module.' prefix
                else:
                    fixed_state_dict[key] = value
            
            disease_model.load_state_dict(fixed_state_dict)
        
        disease_model.eval()
        print("PyTorch model loaded successfully!")
        
    except Exception as torch_error:
        print(f"PyTorch model loading failed: {str(torch_error)}")
        disease_model = None
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Traceback:", traceback.format_exc())
    # Set to None to prevent further errors
    disease_model = None


# Loading crop recommendation model
import os
import sys

print("\n=== Loading Crop Recommendation Model ===")
try:
    # Get the absolute path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    
    print(f"Current directory: {current_dir}")
    print(f"Model directory: {model_dir}")
    
    # List all files in the models directory
    try:
        print("\nContents of models directory:")
        for f in os.listdir(model_dir):
            print(f"- {f}")
    except Exception as e:
        print(f"Could not list models directory: {str(e)}")
    
    # Try to load the model
    # model_filename = 'RandomForest_new.pkl'
    model_filename = 'RandomForest.pkl'
    crop_recommendation_model_path = os.path.join(model_dir, model_filename)
    
    print(f"\nAttempting to load model from: {crop_recommendation_model_path}")
    
    if not os.path.exists(crop_recommendation_model_path):
        raise FileNotFoundError(f"Model file not found at {crop_recommendation_model_path}")
    
    # Check file size
    file_size = os.path.getsize(crop_recommendation_model_path) / (1024 * 1024)  # in MB
    print(f"Model file size: {file_size:.2f} MB")
    
    # Load the model with protocol version handling
    try:
        # Try loading with the default protocol first
        with open(crop_recommendation_model_path, 'rb') as f:
            crop_recommendation_model = pickle.load(f)
    except (pickle.UnpicklingError, AttributeError, ModuleNotFoundError) as e:
        print(f"Error loading model with default protocol: {str(e)}")
        print("Trying with different pickle protocols...")
        
        # Try with different protocols
        for protocol in range(5, 0, -1):
            try:
                with open(crop_recommendation_model_path, 'rb') as f:
                    if protocol >= 4:
                        # For newer Python versions with protocol 4 or 5
                        import pickle5 as pickle_alt
                    else:
                        import pickle as pickle_alt
                    crop_recommendation_model = pickle_alt.load(f)
                    print(f"Successfully loaded model with protocol {protocol}")
                    break
            except Exception as e:
                print(f"Failed with protocol {protocol}: {str(e)}")
        else:
            raise RuntimeError("Could not load model with any protocol")
    
    # Print model information
    print("\nModel loaded successfully!")
    print(f"Model type: {type(crop_recommendation_model)}")
    
    # Try to get some basic model info
    try:
        # Handle different model types
        if hasattr(crop_recommendation_model, 'feature_importances_'):
            print(f"Model has {len(crop_recommendation_model.feature_importances_)} features")
        elif hasattr(crop_recommendation_model, 'coef_'):
            print(f"Model has {crop_recommendation_model.coef_.shape[1]} features")
            
        if hasattr(crop_recommendation_model, 'classes_'):
            print(f"Model has {len(crop_recommendation_model.classes_)} classes")
            print("Class labels:", crop_recommendation_model.classes_)
        elif hasattr(crop_recommendation_model, 'classes'):
            print(f"Model has {len(crop_recommendation_model.classes)} classes")
            print("Class labels:", crop_recommendation_model.classes)
            
        # If this is a scikit-learn pipeline, print its steps
        if hasattr(crop_recommendation_model, 'steps'):
            print("Model is a pipeline with steps:")
            for name, step in crop_recommendation_model.steps:
                print(f"- {name}: {type(step).__name__}")
                
        # If this is a scikit-learn model, print its parameters
        if hasattr(crop_recommendation_model, 'get_params'):
            print("\nModel parameters:")
            params = crop_recommendation_model.get_params()
            for key, value in list(params.items())[:5]:  # Print first 5 params
                print(f"{key}: {value}")
            if len(params) > 5:
                print(f"... and {len(params) - 5} more parameters")
                
    except Exception as e:
        print(f"Could not get model details: {str(e)}")
    
except Exception as e:
    print(f"\n=== ERROR LOADING MODEL ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    print("Traceback:")
    print(traceback.format_exc())
    print("==========================\n")
    
    # Create a dummy model for testing
    from sklearn.ensemble import RandomForestClassifier
    print("\nCreating a dummy model for testing...")
    crop_recommendation_model = RandomForestClassifier()
    # Train on dummy data
    import numpy as np
    X_dummy = np.random.rand(10, 7)  # 7 features
    y_dummy = np.random.randint(0, 5, 10)  # 5 classes
    crop_recommendation_model.fit(X_dummy, y_dummy)
    print("Dummy model created for testing!")


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name: Name of the city to fetch weather for
    :return: tuple of (temperature, humidity) or None if failed
    """
    try:
        if not city_name or not isinstance(city_name, str):
            print("Error: Invalid city name")
            return None
            
        api_key = getattr(config, 'weather_api_key', None)
        if not api_key:
            print("Error: Weather API key not found in config")
            return None
            
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}appid={api_key}&q={city_name}&units=metric"
        
        print(f"Fetching weather for {city_name}...")
        response = requests.get(complete_url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        
        data = response.json()
        
        if data.get("cod") != 200:
            print(f"Weather API error: {data.get('message', 'Unknown error')}")
            return None
            
        main_data = data.get("main", {})
        temperature = main_data.get("temp")
        humidity = main_data.get("humidity")
        
        if temperature is None or humidity is None:
            print("Error: Incomplete weather data received")
            return None
            
        print(f"Weather data - Temp: {temperature}°C, Humidity: {humidity}%")
        return temperature, humidity
        
    except requests.exceptions.RequestException as re:
        print(f"Request error: {str(re)}")
        return None
    except (ValueError, KeyError) as e:
        print(f"Error parsing weather data: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error in weather_fetch: {str(e)}")
        return None


# Mapping from Keras model class names to PyTorch model class names for disease_dic compatibility
KERAS_TO_PYTORCH_MAPPING = {
    'Pepper__bell___Bacterial_spot': 'Pepper,_bell___Bacterial_spot',
    'Pepper__bell___healthy': 'Pepper,_bell___healthy',
    'Potato___Early_blight': 'Potato___Early_blight',
    'Potato___Late_blight': 'Potato___Late_blight',
    'Potato___healthy': 'Potato___healthy',
    'Tomato_Bacterial_spot': 'Tomato___Bacterial_spot',
    'Tomato_Early_blight': 'Tomato___Early_blight',
    'Tomato_Late_blight': 'Tomato___Late_blight',
    'Tomato_Leaf_Mold': 'Tomato___Leaf_Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato___Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot': 'Tomato___Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato___Tomato_mosaic_virus',
    'Tomato_healthy': 'Tomato___healthy'
}

def predict_image(img, model=None):
    """
    Transforms image to tensor and predicts disease label using PlantDNet model
    :params: image (bytes): Image data in bytes
    :return: prediction (string): Predicted disease class
    """
    try:
        # Verify input image
        if not img or len(img) == 0:
            raise ValueError("Empty image data received")
        
        # Use the PlantDNet Keras model (the actual trained model)
        try:
            if hasattr(plant_disease_predictor, 'model') and plant_disease_predictor.model is not None:
                # Save bytes to temporary file for Keras model
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    temp_file.write(img)
                    temp_file_path = temp_file.name
                
                try:
                    # Use the PlantDNet predictor
                    result = plant_disease_predictor.predict_disease(temp_file_path)
                    
                    if result['success']:
                        prediction = result['disease']
                        confidence = result['confidence']
                        
                        # Map Keras class name to PyTorch class name for disease_dic compatibility
                        mapped_prediction = KERAS_TO_PYTORCH_MAPPING.get(prediction, prediction)
                        print(f"PlantDNet Prediction: {prediction} -> Mapped: {mapped_prediction}, Confidence: {confidence:.2f}")
                        return mapped_prediction
                    else:
                        raise ValueError(f"PlantDNet prediction failed: {result.get('error', 'Unknown error')}")
                finally:
                    # Clean up temporary file
                    import os
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
            else:
                raise ValueError("PlantDNet model not available")
                
        except Exception as keras_error:
            print(f"PlantDNet model failed: {str(keras_error)}")
            raise ValueError(f"PlantDNet model prediction failed: {str(keras_error)}")
                
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        raise  # Re-raise to be handled by the calling function

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@app.route('/crop', methods=['GET', 'POST'])
def crop():
    title = 'Harvestify - Crop Recommendation'
    return render_template('standalone_crop.html', title=title)

# Original crop route for backward compatibility
@app.route('/legacy/crop', methods=['GET'])
def legacy_crop():
    title = 'Harvestify - Crop Recommendation (Legacy)'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer-recommend')
def fertilizer_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@app.route('/crop-recommend', methods=['GET', 'POST'])
@app.route('/crop_recommend', methods=['GET', 'POST'])  # Support both GET and POST
@app.route('/crop', methods=['GET'])  # Add a simpler URL for the form
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'
    print("\n=== Received crop predict request ===")
    
    # Handle GET request - show the form
    if request.method == 'GET':
        return render_template('standalone_crop.html', title=title)
    
    try:
        if request.method == 'POST':
            print("Form data received:", request.form)
            
            # Get form data with validation
            try:
                N = int(request.form.get('nitrogen', 0))
                P = int(request.form.get('phosphorous', 0))
                K = int(request.form.get('pottasium', 0))
                ph = float(request.form.get('ph', 7.0))
                rainfall = float(request.form.get('rainfall', 0))
                city = request.form.get('city', '').strip()
                
                print(f"Input values - N: {N}, P: {P}, K: {K}, pH: {ph}, Rainfall: {rainfall}, City: {city}")
                
                if not city:
                    print("Error: No city selected")
                    return render_template('crop.html', 
                                        title=title, 
                                        error='Please select a city')
                
                # Get weather data
                print(f"Fetching weather data for city: {city}")
                weather_data = weather_fetch(city)
                if weather_data is None:
                    print("Error: Could not fetch weather data")
                    return render_template('try_again.html', 
                                        title=title,
                                        error='Could not fetch weather data. Please try again.')
                
                temperature, humidity = weather_data
                print(f"Weather data - Temperature: {temperature}°C, Humidity: {humidity}%")
                
                # Validate input ranges
                if not (0 <= N <= 140) or not (5 <= P <= 145) or not (5 <= K <= 205) or \
                   not (3.5 <= ph <= 9.5) or not (20 <= rainfall <= 300):
                    print("Error: Input values out of range")
                    return render_template('crop.html',
                                        title=title,
                                        error='Please enter values within valid ranges')
                
                # Prepare input data
                input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                print(f"Input data shape: {input_data.shape}")
                print(f"Input data values: {input_data}")
                
                # Make prediction
                try:
                    print("\n=== Model Prediction ===")
                    print(f"Model type: {type(crop_recommendation_model)}")
                    
                    # Check if model has predict method
                    if not hasattr(crop_recommendation_model, 'predict'):
                        raise AttributeError("Model does not have a predict method")
                    
                    # Make prediction
                    prediction = crop_recommendation_model.predict(input_data)
                    print(f"Raw prediction: {prediction}")
                    
                    # Handle different prediction output formats
                    if hasattr(prediction, 'shape') and len(prediction.shape) > 1:
                        prediction = prediction[0]  # Get first prediction if 2D array
                    
                    if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                        final_prediction = prediction[0]  # Get first element if array-like
                    else:
                        final_prediction = str(prediction)
                    
                    print(f"Final prediction: {final_prediction}")
                    
                    return render_template('crop-result.html', 
                                        prediction=final_prediction, 
                                        title=title)
                    
                except Exception as model_error:
                    print("\n=== MODEL PREDICTION ERROR ===")
                    print(f"Error type: {type(model_error).__name__}")
                    print(f"Error message: {str(model_error)}")
                    print("Model type:", type(crop_recommendation_model))
                    print("Model methods:", [method for method in dir(crop_recommendation_model) if not method.startswith('_')])
                    print("Input data type:", type(input_data))
                    print("Input data shape:", input_data.shape if hasattr(input_data, 'shape') else 'No shape')
                    print("Input data dtype:", input_data.dtype if hasattr(input_data, 'dtype') else 'No dtype')
                    print("==========================\n")
                    raise
                
            except ValueError as ve:
                print(f"\n=== VALUE ERROR ===")
                print(f"Error: {str(ve)}")
                print("Traceback:")
                import traceback
                print(traceback.format_exc())
                print("===================\n")
                
                return render_template('crop.html',
                                    title=title,
                                    error=f'Invalid input: {str(ve)}. Please check your input values.')
                
    except Exception as e:
        print("\n=== UNHANDLED ERROR IN CROP PREDICTION ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        import traceback
        print(traceback.format_exc())
        print("===================================\n")
        
        return render_template('crop.html',
                            title=title,
                            error='An unexpected error occurred. Please try again with different values.')

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv(os.path.join(current_dir, '..', 'Data-processed', 'fertilizer.csv'))
    # df = pd.read_csv('C:/Users/DELL/Desktop/Harvestify/Data-raw/FertilizerData.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('disease.html', title=title, error='No file selected')
        
        try:
            # Check file extension
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return render_template('disease.html', title=title, error='Invalid file type. Please upload an image file (PNG, JPG, JPEG)')
            
            img = file.read()
            if not img:
                return render_template('disease.html', title=title, error='Empty file')

            prediction = predict_image(img)
            
            if prediction not in disease_dic:
                return render_template('disease.html', title=title, error='Could not process the image. Please try with another image.')
                
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
            
        except ValueError as ve:
            if "Model Not Loaded" in str(ve):
                return render_template('disease.html', 
                                    title=title, 
                                    error='Prediction Failed: Model Not Loaded')
            print(f"ValueError in disease prediction: {str(ve)}")
            return render_template('disease.html', 
                                title=title, 
                                error=f'Prediction error: {str(ve)}. Please try again.')
        except KeyError as ke:
            print(f"KeyError in disease prediction: {str(ke)}")
            return render_template('disease.html', 
                                title=title, 
                                error=f'Prediction error: {str(ke)}. The model might not recognize this plant disease.')
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error processing image: {error_details}")
            return render_template('disease.html', 
                                title=title, 
                                error=f'Error processing image: {str(e)}. Please try with a different image.')
    
    return render_template('disease.html', title=title, error=None)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5001)
