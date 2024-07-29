import pandas as pd
import numpy as np
import faiss
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import ollama
import torch

# Initialize Ollama client
ollama_client = ollama.Client()

model_name = "multi-qa-mpnet-base-dot-v1"
encoder = SentenceTransformer(model_name)

# Load and preprocess data
def load_data(file_paths):
    data_frames = [pd.read_csv(file) for file in file_paths]
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data


# Index the data using Faiss
def index_data(data, embedding_model, tokenizer):
    dataset = Dataset.from_pandas(data)
    embeddings = encoder.encode(dataset["text"])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


# Retrieve relevant documents
def retrieve_documents(query, index, embedding_model, tokenizer, k=5):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        query_embedding = embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()

    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]


# Generate response using LLM
def generate_response(query, retrieved_documents, llm_model, tokenizer):
    context = " ".join(retrieved_documents)
    inputs = tokenizer(f"Question: {query} Context: {context}", return_tensors='pt', truncation=True, padding=True)
    response = ollama_client.generate(
        model="llama3.1-custom:latest",
        prompt=f"Question: {query} Context: {context}"
    )
    return response


# Main function
def main(file_paths, query):
    # Load data
    data = load_data(file_paths)

    # Load pre-trained models
    embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    llm_model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name)
    llm_model = AutoModel.from_pretrained(llm_model_name)

    # Index the data
    index, embeddings = index_data(data, embedding_model, tokenizer)

    # Retrieve relevant documents
    retrieved_indices = retrieve_documents(query, index, embedding_model, tokenizer)
    retrieved_documents = data.iloc[retrieved_indices][
        'text_column'].tolist()  # Replace 'text_column' with your actual text column

    # Generate response
    response = generate_response(query, retrieved_documents, llm_model, tokenizer)
    print("Response:", response)


# Example usage
file_paths = ["./kusto-data/Ev2RolloutanalyticsEvent.csv"]  # Replace with your actual file paths
query = "What is the capital of France?"
main(file_paths, query)
