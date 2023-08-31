from sentence_transformers import SentenceTransformer
import pickle
import json

# Load the tweets from the pickle file
with open('tweets_text.pkl', 'rb') as f:
    tweets = pickle.load(f)

# Initialize the sentence transformer model
# You can choose a specific model from: https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each tweet
embeddings = model.encode(tweets)

# Create a dictionary to hold the tweet and its corresponding embedding
tweet_embeddings = {tweet: embedding.tolist() for tweet, embedding in zip(tweets, embeddings)}

# Save the tweet and embeddings as a JSON file
with open('tweet_embeddings.json', 'w') as f:
    json.dump(tweet_embeddings, f)

print("Tweet embeddings have been saved to 'tweet_embeddings.json'")
