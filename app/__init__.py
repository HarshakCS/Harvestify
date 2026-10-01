from flask import Flask
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Set a default secret key (in production, use a strong, unique key from environment variables)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_key_please_change_in_production')

# Load configurations
app.config.from_pyfile('config.py', silent=True)

# Ensure the secret key is set after loading config (config.py can override it)
if not app.secret_key:
    app.secret_key = 'dev_key_please_change_in_production'

# Import and register blueprints
from app.routes import routes as routes_blueprint
from app.price_prediction import price_bp as price_prediction_blueprint, init_price_prediction

# Register blueprints
app.register_blueprint(routes_blueprint, url_prefix='/')
# Register price prediction blueprint with /api prefix
app.register_blueprint(price_prediction_blueprint, url_prefix='/api')

# Initialize price prediction module
init_price_prediction(app)

print("\n=== Initializing Application ===")

# Import and load models
try:
    # Import the model loading function
    from app.app_new import load_model, crop_recommendation_model, disease_model
    
    # Load the crop recommendation model if not already loaded
    if crop_recommendation_model is None:
        print("Warning: Could not load crop recommendation model")
    else:
        print("✓ Crop recommendation model loaded successfully")
    
    # Import other components
    try:
        from app.utils.disease import disease_dic
        from app.utils.fertilizer import fertilizer_dic
        from app.utils.model import ResNet9
    except ImportError:
        # Fallback to relative imports if needed
        from utils.disease import disease_dic
        from utils.fertilizer import fertilizer_dic
        from utils.model import ResNet9
        
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    print("Some features may not be available")
