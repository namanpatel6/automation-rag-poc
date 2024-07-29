import os

import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
from glob import glob  # For handling multiple files
import ollama
import requests
from sklearn.model_selection import train_test_split

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

service_endpoint = "https://automation-search-service.search.windows.net"
index_name = "automation-data-cosmos-db-index"
key = "" # *** Insert your Azure Cognitive Search key here ***

search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))

# # Specify the directory containing your CSV files
# csv_directory = "./kusto-data"
# 
# # Collect all CSV file paths
# csv_files = glob(csv_directory + "/Ev2RolloutAnalyticsEvent.csv")
# 
# # Load data from all CSV files, concatenating them
# dfs = []
# for file in csv_files:
#     df = pd.read_csv(file)
#     df.fillna("", inplace=True)  # Replace NaN with empty string
# 
#     # Create a "text" column combining all columns in each row
#     df["text"] = df.apply(
#         lambda row: " ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]),
#         axis=1
#     )
#     dfs.append(df[["text"]])  # Only keep the "text" column
# df = pd.concat(dfs, ignore_index=True)
# 
# # Convert to Hugging Face Dataset
# dataset = Dataset.from_pandas(df)
# 
# # Split data into train and test sets
# # train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
# # train_dataset = Dataset.from_dict(train_data)
# # test_dataset = Dataset.from_dict(test_data)
# 
# # Load sentence transformer model for embeddings
# model_name = "multi-qa-mpnet-base-dot-v1"
# encoder = SentenceTransformer(model_name)
# 
# # Create embeddings (train set only)
# train_embeddings = encoder.encode(dataset["text"])
# 
# # Create FAISS index
# index = faiss.IndexFlatL2(train_embeddings.shape[1])
# index.add(train_embeddings)

# Initialize Ollama client
ollama_client = ollama.Client()

SEARCH_URL = "https://automation-search-service.search.windows.net/indexes/automation-data-cosmos-db-index/docs"


# Retrieve relevant documents
def retrieve_documents(query, top_k=500):
    results = search_client.search(search_text="Service Tree ID is 'f76de4a1-629f-4651-9d76-1d7b56544f3c'", top=top_k, select="BuildNumber", filter="BuildNumber ne null")
    documents = []
    for result in results:
        documents.append("ServiceTreeId: 'f76de4a1-629f-4651-9d76-1d7b56544f3c', BuildNumber: {}".format(result["BuildNumber"]))
    return documents


# Answer questions using RAG and Ollama
def answer_with_rag(query, retrieved_docs):
    answers = []
    try:
        # Customize the prompt for your Kusto data format
        context = "\n".join(retrieved_docs)
        prompt = (f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}\n"
                  f"Just return the answer only without any additional text. Removes duplicates. Use the  ', ' delimeter"
                  f" for separating the results")
        response = ollama_client.generate(
            model="llama3.1-custom:latest",
            prompt=prompt,
            stream=False
        )
        answers.append(
            {
                "answer": response.get('response'),
            }
        )
    except Exception as e:
        print(f"Error generating answer: {e}")
    if answers:
        return max(answers, key=lambda x: x["answer"])
    else:
        return {"answer": "No relevant information found.", "score": 0.0}


# Example usage:
search_query = {
    "search": "Service Tree ID is 'f76de4a1-629f-4651-9d76-1d7b56544f3c'",
    "top": 500,
    "queryType": "simple",
    "select": "BuildNumber",
    "filter": "BuildNumber ne null"
}
query = "Give me a list of build numbers for the service tree id 'f76de4a1-629f-4651-9d76-1d7b56544f3c'"
retrieved_docs = retrieve_documents(search_query)
answer = answer_with_rag(query, retrieved_docs)
print(answer)
