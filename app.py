from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os  # ✅ Import os to access environment variables

app = Flask(__name__)
CORS(app)

# ✅ Load model and scaler
model = load_model('antennae_neuraL_network_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Antenna ML Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0][0]
        return jsonify({'s11_prediction_dB': round(float(prediction), 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Use PORT from environment (Render will set it)
    print(f"Server is running at http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
