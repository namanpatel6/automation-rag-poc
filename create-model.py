from sentence_transformers import SentenceTransformer, models

# 1. Load/Create Your Model
# You canuse pre-trained models from the Sentence Transformers library or fine-tune your own.
model_name = "multi-qa-mpnet-base-dot-v1"  # Replace with your chosen model
model = SentenceTransformer(model_name)  # Load the model


# 2. Save the Model
model.save('embedding-model')  # Save the model to disk