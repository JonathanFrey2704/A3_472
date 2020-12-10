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
    #print(y)

    #tokenize tweets, remove punctuation and stopwords
    tokenized_tweets =[]
    stopwords = set(stopwords.words('english'))
    for tweet in tweets:
        temp = [char for char in tweet if char not in string.punctuation]
        temp = ''.join(temp)
        tokenized_tweets.append([word for word in temp.split() if word not in stopwords])
    #print(tokenized_tweets)

    # make predictions 1 or 0
    y = [1 if q1 == "yes" else 0 for q1 in y]

    vocabulary={} #entire vocabulary
    v1 ={} #vocabulary of 'yes' q1 labels
    v2={} #vocabulary of 'no' q1 labels
    for i, tokenized_tweet in enumerate(tokenized_tweets):
        temp = {}
        if y[i] == 1:
            temp = v1
        else:
            temp = v2

        for word in tokenized_tweet:

            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1

            if word not in temp:
                temp[word] = 1
            else:
                temp[word] += 1

    print(v1)
    print(v2)
    print(vocabulary)



    #priors p(1) and p(0)
    total = len(y)
    p1 = y.count(1)/total
    p0 = y.count(0)/total
    #print(p0)


    #compute conditional probabilities

