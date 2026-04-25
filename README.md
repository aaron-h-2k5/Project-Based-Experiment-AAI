<H3>ENTER YOUR NAME: AARON H</H3>
<H3>ENTER YOUR REGISTER NO.: 212223040001</H3>
<H3>DATE: 25.04.26</H3>
<H1 Align="center">Project Based Experiment<H1>
<H3>Objective:<H3>
To perform sentiment analysis on text data using the NLTK Twitter dataset and count the occurrences of a specific name.
  
<H3>Program:</H3>

```
import nltk
from nltk.corpus import twitter_samples
from nltk.sentiment import SentimentIntensityAnalyzer
import re

nltk.download('twitter_samples')
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

all_tweets = positive_tweets + negative_tweets

all_tweets.extend(custom_tweets)

name_to_count = "Tony Stark"

positive = 0
negative = 0
neutral = 0
name_count = 0

for tweet in all_tweets:
    scores = sid.polarity_scores(tweet)
    
    if scores['compound'] > 0:
        positive += 1
    elif scores['compound'] < 0:
        negative += 1
    else:
        neutral += 1
    
    name_count += len(re.findall(name_to_count, tweet, re.IGNORECASE))

print("Positive Tweets:", positive)
print("Negative Tweets:", negative)
print("Neutral Tweets :", neutral)
print("Occurrences of 'Tony Stark':", name_count)

```
  
<H3>Output:</H3>
<img width="1706" height="313" alt="Screenshot 2026-04-25 at 9 33 06 AM" src="https://github.com/user-attachments/assets/c922d97d-6181-4f76-9e22-20712f76d5a5" />

<H3>Inference:</H3>
Learned how to use NLTK for sentiment analysis, classify text into positive/negative/neutral categories, and extract specific information like name occurrences from text data.
