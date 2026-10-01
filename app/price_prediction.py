from flask import Blueprint, request, jsonify, render_template, current_app
from datetime import datetime
import pandas as pd
import os
import sys
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the predictor
from predict_price import CropPricePredictor

# Load the dataset for dropdown options
def load_dataset():
    """Load the dataset to get unique values for dropdowns"""
    try:
        filepath = os.path.join(project_root, 'Data-processed', 'consolidated_commodity_prices.csv')
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        current_app.logger.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

# Get unique values from dataset
def get_unique_values():
    """Get unique values for dropdowns"""
    df = load_dataset()
    if df.empty:
        return {
            'states': [],
            'districts': {},
            'markets': {},
            'commodities': []
        }
    
    # Get unique states
    states = sorted(df['State'].dropna().unique().tolist())
    
    # Get districts by state
    districts = {}
    for state in states:
        state_districts = df[df['State'] == state]['District'].dropna().unique().tolist()
        districts[state] = sorted(state_districts)
    
    # Get markets by state and district
    markets = {}
    for state, state_df in df.groupby('State'):
        markets[state] = {}
        for district, district_df in state_df.groupby('District'):
            market_list = district_df['Market'].dropna().unique().tolist()
            markets[state][district] = sorted(market_list)
    
    # Get unique commodities
    commodities = sorted(df['Commodity'].dropna().unique().tolist())
    
    return {
        'states': states,
        'markets': markets,
        'commodities': commodities
    }

# Create a Blueprint for price prediction routes
price_bp = Blueprint('price_prediction', __name__, template_folder='templates')

# Global variables to store the predictor and dataset info
predictor = None
dataset_info = {}

def init_price_prediction(app):
    """Initialize the price prediction module with the Flask app"""
    global predictor, dataset_info
    
    try:
        # Initialize predictor
        models_dir = os.path.join(project_root, 'models')
        app.logger.info(f"Initializing price predictor from: {models_dir}")
        
        if not os.path.exists(models_dir):
            app.logger.error(f"Models directory not found: {models_dir}")
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
            
        predictor = CropPricePredictor(models_dir)
        app.logger.info("Price prediction model loaded successfully!")
        
        # Load dataset info for dropdowns
        app.logger.info("Loading dataset info for dropdowns...")
        dataset_info = get_unique_values()
        
        # Log the structure of the loaded data
        app.logger.info(f"Loaded {len(dataset_info.get('states', []))} states")
        app.logger.info(f"Sample states: {dataset_info.get('states', [])[:3]}...")
        
        if 'districts' in dataset_info:
            sample_state = next(iter(dataset_info['districts']), None)
            if sample_state:
                app.logger.info(f"Sample state districts ({sample_state}): {dataset_info['districts'][sample_state][:3]}...")
        
        app.logger.info("Dataset loaded for dropdown options")
        
    except Exception as e:
        app.logger.error(f"Price prediction initialization error: {str(e)}", exc_info=True)
        predictor = None
        dataset_info = {}

@price_bp.route('/price-prediction', methods=['GET'])
def price_prediction_page():
    """Render the price prediction page"""
    return render_template(
        'price_prediction_new.html',
        states=dataset_info.get('states', []),
        commodities=dataset_info.get('commodities', [])
    )

@price_bp.route('/test-prediction', methods=['GET'])
def test_prediction():
    """Test endpoint to verify the prediction model is working"""
    try:
        test_data = {
            'State': 'Karnataka',
            'District': 'Hassan',
            'Market': 'Arakalgud',
            'Commodity': 'Tomato',
            'Variety': 'Tomato',
            'Grade': 'Faq',
            'Arrival_Date': '2023-12-15'
        }
        
        current_app.logger.info("Running test prediction with data: %s", test_data)
        predicted_price = predictor.predict(test_data)
        
        return jsonify({
            'status': 'success',
            'message': 'Test prediction successful',
            'test_data': test_data,
            'predicted_price': predicted_price
        })
        
    except Exception as e:
        current_app.logger.error("Test prediction failed: %s", str(e), exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Test prediction failed',
            'error': str(e)
        }), 500

@price_bp.route('/predict-price', methods=['POST'])
def predict_price():
    """API endpoint for price prediction"""
    current_app.logger.info("Received prediction request")
    
    if predictor is None:
        error_msg = 'Price prediction model is not available'
        current_app.logger.error(error_msg)
        return jsonify({
            'error': error_msg,
            'status': 'error'
        }), 500
    
    try:
        # Get input data from request
        data = request.get_json()
        current_app.logger.info(f"Received data: {data}")
        
        if data is None:
            error_msg = 'No JSON data received in request'
            current_app.logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'status': 'error'
            }), 400
        
        # Validate required fields
        required_fields = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade', 'Arrival_Date']
        for field in required_fields:
            if field not in data or not data[field]:
                error_msg = f'Missing or invalid required field: {field}'
                current_app.logger.error(f"{error_msg}. Data received: {data}")
                return jsonify({
                    'error': error_msg,
                    'status': 'error',
                    'missing_field': field,
                    'received_data': data if current_app.debug else None
                }), 400
        
        try:
            current_app.logger.info(f"Making prediction with data: {data}")
            
            # Log the data types of each field
            for key, value in data.items():
                current_app.logger.info(f"Field '{key}': value='{value}', type={type(value).__name__}")
            
            # Make the prediction
            predicted_price = predictor.predict(data)
            current_app.logger.info(f"Raw prediction result: {predicted_price}, type: {type(predicted_price).__name__}")
            
            # Ensure the prediction is a valid number
            try:
                predicted_price_float = float(predicted_price)
                if predicted_price_float < 0:
                    current_app.logger.warning(f"Predicted price is negative: {predicted_price_float}")
                    predicted_price_float = 0.0  # Ensure non-negative price
                
                # Round to 2 decimal places for currency
                predicted_price_rounded = round(predicted_price_float, 2)
                
                # Format response - ensure all values are JSON serializable
                response = {
                    'status': 'success',
                    'prediction': {
                        'commodity': str(data['Commodity']),
                        'market': str(data['Market']),
                        'district': str(data['District']),
                        'state': str(data['State']),
                        'variety': str(data['Variety']),
                        'grade': str(data['Grade']),
                        'arrival_date': str(data['Arrival_Date']),
                        'predicted_price': predicted_price_rounded,
                        'currency': 'INR',
                        'unit': 'quintal'
                    }
                }
                current_app.logger.info(f"Prediction successful: {response}")
                
                return jsonify(response)
                
            except (ValueError, TypeError) as conversion_error:
                error_msg = f"Failed to convert prediction result to float: {predicted_price}"
                current_app.logger.error(f"{error_msg}. Error: {str(conversion_error)}")
                raise ValueError(error_msg) from conversion_error
                
        except Exception as predict_error:
            current_app.logger.error(f"Prediction failed: {str(predict_error)}", exc_info=True)
            return jsonify({
                'error': 'Failed to make prediction',
                'status': 'error',
                'details': str(predict_error) if current_app.debug else None
            }), 500
        
    except Exception as e:
        current_app.logger.error(f"Unexpected error in predict_price: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred',
            'status': 'error',
            'details': str(e) if current_app.debug else None
        }), 500

# API endpoints for dropdown data
@price_bp.route('/get-commodities', methods=['GET'])
def get_commodities():
    """Get list of unique commodities"""
    try:
        return jsonify({
            'status': 'success',
            'commodities': dataset_info.get('commodities', [])
        })
    except Exception as e:
        current_app.logger.error(f"Error getting commodities: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Failed to load commodities'
        }), 500

@price_bp.route('/get-states', methods=['GET'])
def get_states():
    """Get list of available states (only Karnataka in the dataset)"""
    try:
        return jsonify({
            'status': 'success',
            'states': ['Karnataka']  # Only Karnataka is available in the dataset
        })
    except Exception as e:
        current_app.logger.error(f"Error getting states: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': 'Failed to load states',
            'details': str(e) if current_app.debug else None
        }), 500
@price_bp.route('/get-districts', methods=['GET'])
def get_districts():
    """Get districts for Karnataka (only available state in the dataset)"""
    try:
        current_app.logger.info("=== GET /api/get-districts called ===")
        current_app.logger.info(f"Request args: {request.args}")
        
        state = request.args.get('state', '').strip()
        current_app.logger.info(f"Getting districts for state: '{state}'")
        
        # Hardcoded districts available in the dataset
        available_districts = ['Hassan', 'Mysore', 'Mandya']
        
        # If the state is provided and not Karnataka, return empty list
        if state and state.lower() != 'karnataka':
            current_app.logger.warning(f"No data available for state: '{state}'. Only Karnataka is supported.")
            return jsonify({
                'status': 'success',
                'state': 'Karnataka',
                'districts': []
            })
        
        current_app.logger.info(f"Returning {len(available_districts)} districts for Karnataka: {available_districts}")
        
        response_data = {
            'status': 'success',
            'state': 'Karnataka',
            'districts': available_districts
        }
        
        current_app.logger.info(f"Response data: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error in get_districts: {str(e)}"
        current_app.logger.error(error_msg, exc_info=True)
        return jsonify({
            'status': 'error',
            'error': 'Failed to load districts',
            'details': str(e) if current_app.debug else None
        }), 500
@price_bp.route('/get-markets', methods=['GET'])
def get_markets():
    """Get markets for a given district in Karnataka"""
    try:
        state = request.args.get('state', '').strip()
        district = request.args.get('district', '').strip()
        
        current_app.logger.info(f"Getting markets for state: {state}, district: {district}")
        
        # Define available markets by district
        district_markets = {
            'Hassan': [
                'Arakalgud', 'Arasikere', 'Belur', 'Channarayapatna',
                'Hassan', 'Holenarsipura', 'Sakaleshpura'
            ],
            'Mysore': [
                'Hunsur', 'K.R.Nagar', 'Mysore (Bandipalya)', 'Nanjangud',
                'Piriya Pattana', 'Santhesargur', 'T. Narasipura'
            ],
            'Mandya': [
                'K.R. Pet', 'Maddur', 'Mandya', 'Nagamangala',
                'Pandavapura', 'Srirangapattana'
            ]
        }
        
        # Check if state is Karnataka
        if state.lower() != 'karnataka':
            current_app.logger.warning(f"No data available for state: {state}")
            return jsonify({
                'status': 'success',
                'state': 'Karnataka',
                'district': district,
                'markets': []
            })
        
        # Get markets for the district (case-insensitive match)
        district_lower = district.lower()
        matching_district = next((d for d in district_markets.keys() 
                               if d.lower() == district_lower), None)
        
        if not matching_district:
            current_app.logger.warning(f"No markets found for district: {district}")
            return jsonify({
                'status': 'success',
                'state': 'Karnataka',
                'district': district,
                'markets': []
            })
        
        markets = district_markets[matching_district]
        current_app.logger.info(f"Found {len(markets)} markets for {matching_district}")
        
        return jsonify({
            'status': 'success',
            'state': 'Karnataka',
            'district': matching_district,
            'markets': markets
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting markets: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': 'Failed to load markets',
            'details': str(e) if current_app.debug else None
        }), 500
