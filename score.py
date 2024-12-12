
import joblib
import json
import pandas as pd

def init():
    global model
    # Load the trained model
    model = joblib.load('har_model.pkl')

def run(data):
    try:
        # Parse input data
        input_data = pd.DataFrame(json.loads(data)['data'])
        # Make predictions
        predictions = model.predict(input_data)
        return json.dumps({"predictions": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
