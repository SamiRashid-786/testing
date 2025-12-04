import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# 1. Initialize the Flask app
app = Flask("Assignment-3")

# 2. Load your saved model and preprocessing info
try:
    model = joblib.load('trained_model.joblib')
    model_columns = joblib.load('model_columns.joblib')
    print("Model loaded successfully!")
    print(f"Model expects {len(model_columns)} features")
except FileNotFoundError:
    print("Error: Model files not found. Make sure 'trained_model.joblib' and 'model_columns.joblib' are in the same folder.")
    model = None
    model_columns = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_columns = None

# # 3. Create route to serve the HTML page
# @app.route('/')
# def home():
#     return render_template('index.html')

# 4. Create your prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_columns is None:
        return jsonify({'error': 'Model is not loaded, cannot make prediction.'}), 500

    try:
        # Get data from the 'POST' request
        data = request.get_json()
        
        # Create a DataFrame with all the columns the model expects, filled with 0
        feature_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Fill in the values that were provided in the request
        for key, value in data.items():
            if key in feature_df.columns:
                feature_df[key] = value
            else:
                print(f"Warning: Feature '{key}' not found in model columns")
        
        # Make a prediction
        prediction = model.predict(feature_df)

        # Send the prediction back as JSON
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        # Handle any errors that occur during prediction
        return jsonify({'error': str(e)}), 400

# This line allows you to run the app by just running `python app.py`
if __name__ == '__main__':
    # 'debug=True' means the server will auto-restart when you save changes
    app.run(debug=True)