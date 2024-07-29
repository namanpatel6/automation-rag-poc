import json

from azureml.core import Workspace, Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from sentence_transformers import SentenceTransformer

# 1. Load the SentenceTransformer model 
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')


def init():
    global model  # Ensure the model is globally accessible
    model = SentenceTransformer('./embedding-model')

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']

        # Check if input is a list of strings
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            return {"error": "Invalid input. Please provide a list of strings."}

        # Generate embeddings
        embeddings = model.encode(data)
        return {"data": embeddings.tolist()}  # Convert to list for JSON serialization
    except Exception as e:
        return {"error": str(e)}
