import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the Flask app
from app import app

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, port=5001, host='0.0.0.0')
