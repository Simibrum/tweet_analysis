import json
import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from datetime import datetime

# Load the tweets
with open('tweets.js', 'r', encoding='utf-8') as f:
    data = f.read()
    json_data = json.loads(data[data.index('['):])

# Extract tweet text and created_at fields
tweets = []
timestamps = []
for entry in json_data:
    tweet_text = entry["tweet"]["full_text"]
    created_at = entry["tweet"]["created_at"]
    tweets.append(tweet_text)
    timestamps.append(created_at)

# Create a DataFrame
df = pd.DataFrame({'tweet': tweets, 'created_at': timestamps})
df['created_at'] = pd.to_datetime(df['created_at'])
df.sort_values('created_at', inplace=True)

# Calculate sentiment scores
df['sentiment'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Resample the data by month and take the mean
df.set_index('created_at', inplace=True)
df_resampled = df['sentiment'].resample('M').mean()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df_resampled.index, df_resampled.values, marker='o')
plt.title('Sentiment Over Time')
plt.xlabel('Time')
plt.ylabel('Sentiment')
plt.grid(True)
plt.show()