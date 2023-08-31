import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import textwrap

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load tweet embeddings and texts
data = np.load('tweet_embeddings.npz')
tweet_texts = data['texts']
tweet_embeddings = data['embeddings']

# Get user input for query text and number of results
query_text = input("Enter the query text: ")
num_results = int(input("Enter the number of results to return: "))

# Encode the query text
query_vector = model.encode([query_text])
query_vector = query_vector.reshape(1, -1)

# Compute cosine similarities
similarities = cosine_similarity(query_vector, tweet_embeddings)

# Sort by similarity
sorted_indices = np.argsort(-similarities[0])

# Print top N most similar tweets
print(f"Top {num_results} most similar tweets to '{query_text}':")
for i in range(num_results):
    index = sorted_indices[i]
    wrapped_text = textwrap.fill(tweet_texts[index], width=200)
    print(f"{i + 1}. {wrapped_text}\n(Similarity: {similarities[0][index]:.4f})\n")
