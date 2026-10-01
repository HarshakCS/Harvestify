import sys
import os
import io
import logging

# Set up UTF-8 encoding for stdout and stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Create a unique log file name with timestamp to avoid conflicts
import time
log_file = os.path.join(log_dir, f'app_{int(time.time())}.log')

# Configure logging to both console and file
handlers = [logging.StreamHandler(sys.stdout)]

try:
    # Try to add file handler, but don't fail if we can't
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    handlers.append(file_handler)
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG for more detailed logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
except Exception as e:
    # If we can't log to file, just use console logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not set up file logging: {e}. Using console logging only.")
else:
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the Flask app
from flask_cors import CORS
from app import app

# Enable CORS for all routes during development
# WARNING: This allows all origins. In production, restrict this to specific domains.
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

if __name__ == "__main__":
    try:
        # Set environment variables
        os.environ['FLASK_DEBUG'] = '1'
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the application
        app.run(debug=True, port=5001, host='0.0.0.0')
    except Exception as e:
        logging.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise
