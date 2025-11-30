import os
from joblib import load

# Load model
def load_model(model_name: str):
    path = os.path.join("models", model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return load(path)
