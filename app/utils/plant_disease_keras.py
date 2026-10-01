"""
Plant Disease Prediction Module using PlantDNet.h5 Keras Model
Direct integration with the working Plant-Disease-Prediction app
"""
import os
import sys
import numpy as np
from huggingface_hub import hf_hub_download

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Disease classes for PlantDNet model
PLANTDNET_DISEASE_CLASSES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

class PlantDiseasePredictor:
    """Plant disease prediction using PlantDNet.h5 model with PyTorch fallback"""
    
    def __init__(self):
        # self.model_path = os.path.join(project_root, 'Plant-Disease-Prediction', 'PlantDNet.h5')
        self.model_path = hf_hub_download(repo_id="HARSHAKCS/harvestify-models",
        filename="PlantDNet.h5"
        )
        self.disease_classes = PLANTDNET_DISEASE_CLASSES
        self.model_available = self.check_model_availability()
        self.model = None
        self.fallback_model = None
        self.use_fallback = False
        
        # Try to load primary model first
        self.load_model()
        
        # If primary model failed, load fallback
        if self.model is None:
            print("Primary model failed, loading fallback...")
            self.load_fallback_model()
        
    def check_model_availability(self):
        """Check if model file exists"""
        exists = os.path.exists(self.model_path)
        if exists:
            print(f"PlantDNet model found at: {self.model_path}")
        else:
            print(f"PlantDNet model not found at: {self.model_path}")
        return exists
    
    def load_model(self):
        """Load PlantDNet model using the same approach as the original working app"""
        if not self.model_available:
            print("Model file not available")
            return
            
        try:
            # Change to Plant-Disease-Prediction directory
            original_cwd = os.getcwd()
            plant_disease_dir = os.path.join(project_root, 'Plant-Disease-Prediction')
            os.chdir(plant_disease_dir)
            
            try:
                import tensorflow as tf
                from tensorflow.keras.preprocessing import image
                
                # Disable oneDNN warnings
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
                
                # Load model exactly like the original app
                print("Loading PlantDNet model...")
                self.model = tf.keras.models.load_model('PlantDNet.h5', compile=False)
                self.tf = tf
                self.image = image
                print("+ PlantDNet model loaded successfully!")
                print("+ PlantDNet model ready for predictions!")
                return
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                import traceback
                print("Traceback:", traceback.format_exc())
                self.model = None
            finally:
                # Change back to original directory
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"Critical error in load_model: {str(e)}")
            self.model = None
    
    def load_fallback_model(self):
        """Load PyTorch ResNet9 model from specific path"""
        try:
            import torch
            from torchvision import transforms
            from PIL import Image
            import sys
            
            # Add app directory to path to import ResNet9
            app_dir = os.path.join(project_root, 'app')
            if app_dir not in sys.path:
                sys.path.insert(0, app_dir)
            
            from utils.model import ResNet9
            
            # Use the specific path provided by user
            model_path = r"B:\Harvestify - Copy\models\plant_disease_model.pth"
            
            if os.path.exists(model_path):
                print(f"Loading PyTorch fallback model from: {model_path}")
                
                # Initialize ResNet9 with 38 classes
                self.fallback_model = ResNet9(3, 38)
                
                # Load state dict
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Handle different save formats
                if isinstance(state_dict, dict):
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                
                # Try to load state dict directly first
                try:
                    self.fallback_model.load_state_dict(state_dict)
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
                    
                    self.fallback_model.load_state_dict(fixed_state_dict)
                
                self.fallback_model.eval()
                
                # Setup transforms for PyTorch model
                self.pytorch_transform = transforms.Compose([
                    transforms.Resize((256, 256)),  # Use larger input for ResNet9 architecture
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Update disease classes to match PyTorch model (38 classes)
                self.disease_classes = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]
                
                self.use_fallback = True
                print("+ PyTorch ResNet9 fallback model loaded successfully!")
                print(f"+ Using fallback model with {len(self.disease_classes)} disease classes")
                
            else:
                print(f"PyTorch fallback model not found at: {model_path}")
                
        except Exception as e:
            print(f"Error loading fallback model: {str(e)}")
            import traceback
            print("Traceback:", traceback.format_exc())
            self.fallback_model = None
    
    def preprocess_image(self, image_path):
        """Preprocess image for PlantDNet model prediction"""
        try:
            if self.model is None:
                return None
                
            # Load and preprocess image (64x64 as per original model)
            # Remove grayscale parameter as it's deprecated in newer TF versions
            img = self.image.load_img(image_path, target_size=(64, 64))
            x = self.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = np.array(x, 'float32')
            x /= 255
            return x
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict_disease(self, image_path):
        """
        Predict disease from plant leaf image using either primary or fallback model
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results with disease class and confidence
        """
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'Image file not found: {image_path}',
                'disease': None,
                'confidence': 0
            }
        
        try:
            if self.use_fallback and self.fallback_model is not None:
                # Use PyTorch fallback model
                return self._predict_with_fallback(image_path)
            elif self.model is not None:
                # Use primary TensorFlow model
                return self._predict_with_primary(image_path)
            else:
                return {
                    'success': False,
                    'error': 'No model available',
                    'disease': None,
                    'confidence': 0
                }
                
        except Exception as e:
            print(f"Error in predict_disease: {str(e)}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'disease': None,
                'confidence': 0
            }
    
    def _predict_with_primary(self, image_path):
        """Predict using primary TensorFlow model"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return {
                    'success': False,
                    'error': 'Failed to preprocess image',
                    'disease': None,
                    'confidence': 0
                }
            
            # Make prediction
            preds = self.model.predict(processed_image)
            predicted_index = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))
            disease_class = self.disease_classes[predicted_index]
            
            return {
                'success': True,
                'disease': disease_class,
                'confidence': confidence,
                'model_used': 'PlantDNet (TensorFlow)'
            }
            
        except Exception as e:
            print(f"Error in primary prediction: {str(e)}")
            return {
                'success': False,
                'error': f'Primary model prediction failed: {str(e)}',
                'disease': None,
                'confidence': 0
            }
    
    def _predict_with_fallback(self, image_path):
        """Predict using fallback PyTorch model"""
        try:
            import torch
            from PIL import Image
            
            # Load and preprocess image for PyTorch model
            image = Image.open(image_path).convert('RGB')
            
            # Debug: print image size
            print(f"Original image size: {image.size}")
            
            # Resize to 256x256 (Required for ResNet9 with MaxPool2d(4))
            image = image.resize((256, 256))
            image_tensor = self.pytorch_transform(image).unsqueeze(0)
            
            # Debug: print tensor shape
            print(f"Tensor shape: {image_tensor.shape}")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.fallback_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_index = predicted.item()
                confidence = confidence.item()
                disease_class = self.disease_classes[predicted_index]
            
            return {
                'success': True,
                'disease': disease_class,
                'confidence': confidence,
                'model_used': 'ResNet9 (PyTorch Fallback)'
            }
            
        except Exception as e:
            print(f"Error in fallback prediction: {str(e)}")
            import traceback
            print("Traceback:", traceback.format_exc())
            return {
                'success': False,
                'error': f'Fallback model prediction failed: {str(e)}',
                'disease': None,
                'confidence': 0
            }
    
    def get_disease_info(self, disease_class):
        """Get detailed information about the disease"""
        try:
            # Import disease dictionary from utils
            from app.utils.disease import disease_dic
            return disease_dic.get(disease_class, "No detailed information available for this disease.")
        except ImportError:
            return "Disease information not available."

# Global predictor instance
plant_disease_predictor = PlantDiseasePredictor()

def predict_plant_disease(image_path):
    """
    Convenience function for plant disease prediction
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Prediction results
    """
    return plant_disease_predictor.predict_disease(image_path)

def get_disease_details(disease_class):
    """
    Get detailed information about a disease
    
    Args:
        disease_class (str): Disease class name
        
    Returns:
        str: Detailed disease information
    """
    return plant_disease_predictor.get_disease_info(disease_class)
