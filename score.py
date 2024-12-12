import joblib
import json
import pandas as pd

# Initialize global models dictionary
models = {}

def init():
    global models
    # Load both trained models
    models['iteration1'] = joblib.load('har_model.pkl')
    models['iteration2'] = joblib.load('random_forest_model_v2.pkl')

def run(data):
    try:
        # Parse input data
        request = json.loads(data)
        model_version = request.get('model_version', 'iteration1')  # Default to iteration1
        input_data = pd.DataFrame(request['data'])

        # Check if the requested model version exists
        if model_version not in models:
            raise ValueError(f"Model version '{model_version}' not found. Available versions: {list(models.keys())}")

        # Get the corresponding model
        model = models[model_version]

        # Make predictions
        predictions = model.predict(input_data)
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
