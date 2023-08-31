import json
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Load tweets from the JSON file
with open("tweets.js", "r") as f:
    data_str = f.read()[len("window.YTD.tweet.part0 = "):]
    tweets_data = json.loads(data_str)

# Extract tweet texts and created_at fields
tweets = [(tweet['tweet']['created_at'], tweet['tweet']['full_text']) for tweet in tweets_data]

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(tweets, columns=['created_at', 'text'])

# Convert 'created_at' to datetime format and extract day and month
df['created_at'] = pd.to_datetime(df['created_at'])
df['day_of_week'] = df['created_at'].dt.day_name()
df['month'] = df['created_at'].dt.month_name()

# Calculate sentiment scores
df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Group by day of the week and month, then calculate mean sentiment
df_by_day = df.groupby('day_of_week')['sentiment'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
df_by_month = df.groupby('month')['sentiment'].mean().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])

# Plotting
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.title('Average Sentiment by Day of the Week')
df_by_day.plot(kind='bar')
plt.xlabel('Day of the Week')
plt.ylabel('Average Sentiment')

plt.subplot(1, 2, 2)
plt.title('Average Sentiment by Month')
df_by_month.plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Average Sentiment')

plt.tight_layout()
plt.show()