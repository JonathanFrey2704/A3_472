import csv
import string
from nltk.corpus import stopwords

if __name__ == '__main__':

    #read data
    tsv_file = open("covid_training.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    next(read_tsv) #ignore header

    tweet_ids = []
    tweets = []
    y = []
    for row in read_tsv:
        tweet_ids.append(row[0])
        tweets.append(row[1].lower())
        y.append(row[2])
    #print(tweets)

    #tokenize tweets, remove punctuation and stopwords
    tokenized_tweets =[]
    stopwords = set(stopwords.words('english'))
    for tweet in tweets:
        temp = [char for char in tweet if char not in string.punctuation]
        temp = ''.join(temp)
        tokenized_tweets.append([word for word in temp.split() if word not in stopwords])
    #print(tokenized_tweets)

    vocabulary ={}
    for tokenized_tweet in tokenized_tweets:
        for word in tokenized_tweet:
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1

    #print(vocabulary)

