import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the tweet JSON data
with open('tweets.js', 'r', encoding='utf-8') as f:
    raw_data = f.read()
    json_data = json.loads(raw_data[raw_data.index('['):])

# Extract the tweet timestamps
timestamps = [tweet['tweet']['created_at'] for tweet in json_data]

# Create a DataFrame
df = pd.DataFrame(timestamps, columns=['timestamp'])

# Convert to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the timestamp as the index
df.set_index('timestamp', inplace=True)

# Resample and count for each month
df_resampled = df.resample('M').size()

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(df_resampled.index, df_resampled.values, marker='o')
plt.title('Tweet Frequency Over Time')
plt.xlabel('Time')
plt.ylabel('Number of Tweets')
plt.grid(True)
plt.show()