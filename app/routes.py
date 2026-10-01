from flask import Blueprint, render_template, request, redirect, url_for, jsonify, flash
from markupsafe import Markup
import os
import sys
import numpy as np
import pandas as pd
import json
import traceback

# Import the model and utility functions from app_new
from app.app_new import crop_recommendation_model, weather_fetch, predict_image
from app.utils.chatbot import chatbot

# Create a Blueprint for routes
routes = Blueprint('routes', __name__)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import config
try:
    from app import config
except ImportError:
    import config

# Import local modules
try:
    from app.utils.disease import disease_dic
    from app.utils.fertilizer import fertilizer_dic
except ImportError:
    # Fallback to relative imports if absolute imports fail
    from utils.disease import disease_dic
    from utils.fertilizer import fertilizer_dic

# ===============================================================================================
# ROUTES
# ===============================================================================================

@routes.route('/')
def home():
    """Render the home page"""
    try:
        title = 'Harvestify - Home'
        return render_template('index.html', title=title)
    except Exception as e:
        print(f"Error in home route: {str(e)}")
        print(traceback.format_exc())
        return render_template('error.html', error="An error occurred while loading the home page."), 500

@routes.route('/crop', methods=['GET', 'POST'])
@routes.route('/crop-prediction', methods=['GET', 'POST'])
def crop():
    title = 'Harvestify - Crop Recommendation'
    
    if request.method == 'POST':
        # Get form data with default values
        nitrogen = request.form.get('nitrogen', '0')
        phosphorous = request.form.get('phosphorous', '0')
        pottasium = request.form.get('potassium', '0')
        temperature = request.form.get('temperature', '25.0')
        humidity = request.form.get('humidity', '70.0')
        ph = request.form.get('ph', '7.0')
        rainfall = request.form.get('rainfall', '100')
        
        # Convert empty strings to defaults
        nitrogen = float(nitrogen) if nitrogen != '' else 0
        phosphorous = float(phosphorous) if phosphorous != '' else 0
        pottasium = float(pottasium) if pottasium != '' else 0
        temperature = float(temperature) if temperature != '' else 25.0
        humidity = float(humidity) if humidity != '' else 70.0
        ph = float(ph) if ph != '' else 7.0
        rainfall = float(rainfall) if rainfall != '' else 100.0
        
        # Process the form data and make predictions
        try:
            # Convert inputs to float with validation
            try:
                nitrogen = float(nitrogen)
                phosphorous = float(phosphorous)
                pottasium = float(pottasium)
                ph = float(ph)
                rainfall = float(rainfall)
            except ValueError as ve:
                flash('Please enter valid numeric values for all fields.', 'error')
                return render_template('standalone_crop.html', 
                                    title=title,
                                    nitrogen=nitrogen,
                                    phosphorous=phosphorous,
                                    pottasium=pottasium,
                                    ph=ph,
                                    rainfall=rainfall,
                                    city=city,
                                    sts=state)
            
            # Prepare input data for prediction
            # The model expects features in this order: N, P, K, temperature, humidity, ph, rainfall
            input_data = np.array([[
                nitrogen,
                phosphorous,
                pottasium,
                temperature,
                humidity,
                ph,
                rainfall
            ]])
            
            # Make prediction using the loaded model
            prediction = None
            recommendations = []
            
            if crop_recommendation_model is not None:
                try:
                    # Make prediction
                    prediction = crop_recommendation_model.predict(input_data)[0]
                    
                    # Get prediction probabilities if available
                    if hasattr(crop_recommendation_model, 'predict_proba'):
                        proba = crop_recommendation_model.predict_proba(input_data)[0]
                        # Get top 3 predictions with their probabilities
                        top_n = min(3, len(proba))
                        top_indices = np.argsort(proba)[-top_n:][::-1]
                        
                        for idx in top_indices:
                            crop_name = crop_recommendation_model.classes_[idx]
                            confidence = float(proba[idx])
                            recommendations.append({
                                'name': crop_name,
                                'confidence': confidence,
                                'image_path': f"/static/images/crops/{crop_name.lower().replace(' ', '_')}.jpg"
                            })
                    else:
                        # If no probabilities, just return the single prediction with 100% confidence
                        recommendations.append({
                            'name': prediction,
                            'confidence': 1.0,
                            'image_path': f"/static/images/crops/{prediction.lower().replace(' ', '_')}.jpg"
                        })
                    
                    # For AJAX requests, return JSON
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({
                            'status': 'success',
                            'recommendations': recommendations,
                            'message': 'Here are the top crop recommendations based on your soil and weather conditions.'
                        })
                    
                    # For regular form submission, render template with results
                    return render_template('crop_new.html', 
                                        title=title,
                                        recommendations=recommendations,
                                        nitrogen=nitrogen,
                                        phosphorous=phosphorous,
                                        potassium=pottasium,
                                        temperature=temperature,
                                        humidity=humidity,
                                        ph=ph,
                                        rainfall=rainfall)
                    
                except Exception as pe:
                    error_msg = f'Error making prediction: {str(pe)}'
                    print(f"\n=== PREDICTION ERROR ===\n{error_msg}\n{traceback.format_exc()}\n======================")
                    
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return jsonify({
                            'status': 'error',
                            'message': error_msg
                        }), 400
                    
                    flash(error_msg, 'error')
            else:
                error_msg = 'Prediction model not available. Please try again later.'
                print("\n=== MODEL NOT LOADED ===\n")
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'status': 'error',
                        'message': error_msg
                    }), 503
                
                flash(error_msg, 'error')
                
            # If we get here, there was an error and we're not using AJAX
            return render_template('crop_new.html', 
                                 title=title,
                                 nitrogen=nitrogen,
                                 phosphorous=phosphorous,
                                 potassium=pottasium,
                                 temperature=temperature,
                                 humidity=humidity,
                                 ph=ph,
                                 rainfall=rainfall)
                                
        except Exception as e:
            print(f"Error in crop prediction: {str(e)}\n{traceback.format_exc()}")
            flash('An unexpected error occurred. Please try again.', 'error')
            return render_template('standalone_crop.html', 
                                title=title,
                                nitrogen=nitrogen,
                                phosphorous=phosphorous,
                                pottasium=pottasium,
                                ph=ph,
                                rainfall=rainfall,
                                city=city,
                                sts=state)
    
    # For GET requests, just render the form with default values
    return render_template('crop_new.html', 
                         title=title,
                         nitrogen='',
                         phosphorous='',
                         potassium='',
                         temperature='',
                         humidity='',
                         ph='',
                         rainfall='')

@routes.route('/fertilizer', methods=['GET', 'POST'])
@routes.route('/fertilizer-recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    """Handle fertilizer recommendation requests"""
    title = 'Harvestify - Fertilizer Recommendation'
    
    if request.method == 'POST':
        # Get form data with default values
        temperature = request.form.get('temperature', '25.0')
        humidity = request.form.get('humidity', '70.0')
        soil_type = request.form.get('soil_type', '')
        crop_type = request.form.get('crop_type', '')
        nitrogen = request.form.get('nitrogen', '0')
        phosphorous = request.form.get('phosphorous', '0')
        potassium = request.form.get('potassium', '0')
        ph = request.form.get('ph', '7.0')
        
        # Convert to appropriate types
        try:
            temperature = float(temperature) if temperature else 25.0
            humidity = float(humidity) if humidity else 70.0
            nitrogen = float(nitrogen) if nitrogen else 0
            phosphorous = float(phosphorous) if phosphorous else 0
            potassium = float(potassium) if potassium else 0
            ph = float(ph) if ph else 7.0
        except ValueError:
            error_msg = 'Please enter valid numeric values for all fields.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'status': 'error',
                    'message': error_msg
                }), 400
            flash(error_msg, 'error')
            return render_template('fertilizer_new.html', title=title)
        
        # Generate fertilizer recommendation based on input values
        try:
            # This is a simple rule-based recommendation - replace with your ML model
            recommendations = []
            
            # Check nutrient levels and recommend fertilizers
            if nitrogen < 30:
                recommendations.append({
                    'name': 'Urea',
                    'type': 'Nitrogen-rich',
                    'npk_ratio': '46-0-0',
                    'confidence': 0.9 if nitrogen < 15 else 0.7
                })
            elif nitrogen > 70:
                recommendations.append({
                    'name': 'No nitrogen needed',
                    'type': 'Information',
                    'message': 'Your soil has sufficient nitrogen levels.',
                    'confidence': 0.9
                })
                
            if phosphorous < 30:
                recommendations.append({
                    'name': 'DAP',
                    'type': 'Phosphorous-rich',
                    'npk_ratio': '18-46-0',
                    'confidence': 0.85 if phosphorous < 20 else 0.65
                })
                
            if potassium < 30:
                recommendations.append({
                    'name': 'MOP',
                    'type': 'Potassium-rich',
                    'npk_ratio': '0-0-60',
                    'confidence': 0.88 if potassium < 20 else 0.68
                })
                
            # Add crop-specific recommendations
            crop_specific = []
            if crop_type.lower() in ['rice', 'wheat', 'maize']:
                crop_specific.append({
                    'name': 'NPK Complex',
                    'type': 'Balanced',
                    'npk_ratio': '10-26-26',
                    'confidence': 0.95,
                    'reason': f'Recommended for {crop_type} during early growth stages.'
                })
            
            # Combine recommendations
            if recommendations or crop_specific:
                # Sort by confidence (highest first)
                all_recommendations = sorted(
                    recommendations + crop_specific,
                    key=lambda x: x['confidence'],
                    reverse=True
                )
                
                # For AJAX requests, return JSON
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'status': 'success',
                        'fertilizer': all_recommendations[0],  # Top recommendation
                        'all_recommendations': all_recommendations,
                        'confidence': all_recommendations[0]['confidence'],
                        'instructions': [
                            f"Apply {all_recommendations[0]['name']} at a rate of 100-150 kg per acre.",
                            "Incorporate the fertilizer into the top 2-4 inches of soil.",
                            "Water the field after application to help nutrients reach the root zone."
                        ]
                    })
                
                # For regular form submission, render template with results
                return render_template('fertilizer_new.html',
                                    title=title,
                                    fertilizer=all_recommendations[0],
                                    all_recommendations=all_recommendations,
                                    temperature=temperature,
                                    humidity=humidity,
                                    soil_type=soil_type,
                                    crop_type=crop_type,
                                    nitrogen=nitrogen,
                                    phosphorous=phosphorous,
                                    potassium=potassium,
                                    ph=ph)
            else:
                message = "Your soil has balanced nutrient levels. No additional fertilizers are needed at this time."
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({
                        'status': 'success',
                        'message': message,
                        'fertilizer': None
                    })
                
                return render_template('fertilizer_new.html',
                                    title=title,
                                    message=message,
                                    temperature=temperature,
                                    humidity=humidity,
                                    soil_type=soil_type,
                                    crop_type=crop_type,
                                    nitrogen=nitrogen,
                                    phosphorous=phosphorous,
                                    potassium=potassium,
                                    ph=ph)
                                    
        except Exception as e:
            error_msg = f'Error generating fertilizer recommendation: {str(e)}'
            print(f"\n=== FERTILIZER RECOMMENDATION ERROR ===\n{error_msg}\n{traceback.format_exc()}\n===================================")
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'status': 'error',
                    'message': error_msg
                }), 500
            
            flash(error_msg, 'error')
            return render_template('fertilizer_new.html', title=title)
    
    # For GET requests, just render the form with default values
    return render_template('fertilizer_new.html',
                         title=title,
                         temperature='',
                         humidity='',
                         soil_type='',
                         crop_type='',
                         nitrogen='',
                         phosphorous='',
                         potassium='',
                         ph='')

@routes.route('/disease-prediction', methods=['GET', 'POST'])
def disease_prediction():
    """Handle disease prediction requests"""
    try:
        title = 'Harvestify - Disease Prediction'
        
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)
                
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
                
            if file:
                try:
                    # Create uploads directory if it doesn't exist
                    upload_dir = os.path.join('app', 'static', 'uploads')
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    # Generate a unique filename
                    from werkzeug.utils import secure_filename
                    import uuid
                    
                    # Get file extension
                    file_ext = os.path.splitext(file.filename)[1]
                    # Create a unique filename
                    filename = f"{uuid.uuid4().hex}{file_ext}"
                    
                    # Save the file
                    img_path = os.path.join(upload_dir, filename)
                    file.save(img_path)
                    
                    print(f"\n=== Processing image: {img_path} ===")
                    
                    # Make prediction
                    prediction, confidence = predict_image(img_path)
                    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
                    
                    # Get disease info
                    disease_info = disease_dic.get(prediction, "No detailed information available for this disease.")
                    
                    # Format the prediction for display
                    if '___' in prediction:
                        plant, disease = prediction.split('___', 1)
                        disease = disease.replace('_', ' ').title()
                    else:
                        plant = "Unknown"
                        disease = prediction.replace('_', ' ').title()
                    
                    # Create web-accessible path
                    web_img_path = f"uploads/{filename}"
                    
                    return render_template('disease.html', 
                                        title=title, 
                                        prediction=prediction,
                                        plant=plant,
                                        disease=disease,
                                        confidence=f"{confidence*100:.1f}%",
                                        disease_info=disease_info,
                                        img_path=web_img_path)
                                        
                except Exception as e:
                    error_msg = f"Error processing image: {str(e)}"
                    print(error_msg)
                    flash(error_msg, 'error')
                    return redirect(request.url)
        
        # For GET requests or if no file was uploaded
        return render_template('disease.html', title=title)
        
    except Exception as e:
        error_msg = f"Error in disease_prediction route: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        flash('An error occurred while processing your request.', 'error')
        return render_template('disease.html', title=title)

@routes.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages using Ollama's phi3 model."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get system prompt from the request or use a default one
        system_prompt = data.get('system_prompt', "You are a helpful AI assistant for farmers. Provide concise and accurate information about farming, crops, weather, and related agricultural topics.")
        
        # Generate response using the chatbot
        response = chatbot.generate_response(message, system_prompt)
        
        return jsonify({
            'response': response,
            'model': chatbot.model_name
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@routes.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear the chat history."""
    try:
        chatbot.clear_history()
        return jsonify({'status': 'Chat history cleared'})
    except Exception as e:
        print(f"Error clearing chat history: {str(e)}")
        return jsonify({'error': 'Failed to clear chat history'}), 500
