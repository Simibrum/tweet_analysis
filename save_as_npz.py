import json
import numpy as np

# Load the tweet embeddings from the JSON file
with open('tweet_embeddings.json', 'r') as f:
    tweet_embeddings = json.load(f)

# Extract tweet texts and embeddings
tweet_texts = list(tweet_embeddings.keys())
embeddings = np.array([tweet_embeddings[tweet] for tweet in tweet_texts])

# Save as compressed NumPy file
np.savez_compressed('tweet_embeddings.npz', texts=tweet_texts, embeddings=embeddings)

print("Tweet embeddings have been saved to 'tweet_embeddings.npz'")
