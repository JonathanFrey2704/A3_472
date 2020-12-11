import csv
import string
from nltk.corpus import stopwords
from math import log10
from output import out
from textProcess import testData
if __name__ == '__main__':

    #read data
    tsv_file = open("covid_training.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    next(read_tsv) #ignore header

    tweets = {} #dictionary of tweets. tweet_id -> tweet content
    y = {}  # dictionary of ground truths. tweet_id -> "yes" or "no"
    for row in read_tsv:
        tweets[row[0]] = row[1].lower()
        y[row[0]] = row[2]
    #tokenize tweets, remove punctuation and stopwords
    tokenized_tweets =[]
    tweet_ids = []
    stopwords = set(stopwords.words('english'))
    for tweet_id, tweet in tweets.items():
        # temp = [char for char in tweet if char not in string.punctuation]
        # temp = ''.join(temp)
        # tokenized_tweets.append([word for word in temp.split() if word not in stopwords])
        tokens = tweet.split(' ')
        tokenized_tweets.append(tokens)
        tweet_ids.append(tweet_id)

    y_int = [1 if y[t_id] == "yes" else 0 for t_id in y.keys()] # make prediction array for every tweet, used for priors

    vocabulary={} #entire vocabulary
    v1 ={} #vocabulary of 'yes' q1 labels
    v2={} #vocabulary of 'no' q1 labels
    for i, tokenized_tweet in enumerate(tokenized_tweets):
        temp = {}
        if y[tweet_ids[i]] == 'yes':
            label = True
        else:
            label = False

        for word in tokenized_tweet:

            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1

            if label:
                v1[word] = 1 if word not in v1.keys() else v1[word]+1
            else:
               v2[word] = 1 if word not in v2.keys() else v2[word]+1 


    #UNCOMMENT TO RUN WITH FILTERED VOCABULARY
    #filtered vocabulary
    # for k in list(vocabulary.keys()):
    #     if vocabulary[k] == 1:
    #         vocabulary.pop(k)
    #         if k in v1.keys():
    #             v1.pop(k)
    #         else:
    #             v2.pop(k)


    #priors p(1) and p(0)
    total = len(y.keys())
    p1 = y_int.count(1)/total
    p2 = y_int.count(0)/total
    cond_yes = {} #prior probs for 'yes' class
    cond_no = {} #prior probs for 'no' class

    total_c1 = 0
    for tf in v1.values(): #get total word count for 'yes' class
        total_c1 += tf

    total_c2 = 0
    for tf in v2.values(): #get total word count for 'no' class
        total_c2 += tf

    V = len(vocabulary.keys()) # size of vocabulary
    d = 0.01 # smoothing delta

    for term in vocabulary.keys(): # compute conditional probs for 'yes' and 'no' classes
        tf_c1 = v1[term] if term in v1.keys() else 0 # set term freq. to 0 if word not in our class vocab
        cond_yes[term] = (tf_c1 + d)/(total_c1 + d*V) # smoothing from spam filtering slides, slide 9

        tf_c2 = v2[term] if term in v2.keys() else 0
        cond_no[term] = (tf_c2 + d)/(total_c2 + d*V)

    classifications = {} # dictionary to store predicted label. tweet_id -> "yes" or "no"
    score = {}

    x_test, y_test = testData('covid_test_public.tsv') #test data

    for tweet_id, tweet in x_test.items():
        tokens = tweet.split(' ')
        c1 = log10(p1)
        c2 = log10(p2)

        for term in tokens:
            if term in stopwords:
                continue
            if term not in vocabulary.keys():
                continue
            c1 += log10(cond_yes[term])
            c2 += log10(cond_no[term])

        if c1 > c2: # if class "yes" is more likely than class "no"
            classifications[tweet_id] = "yes" 
            score[tweet_id] = c1
        elif c1 < c2:
            classifications[tweet_id] = "no"
            score[tweet_id] = c2
        else:
            classifications[tweet_id] = "maybe" # this is me trying to be clever
            score[tweet_id] = c1

    labels = {}
    #final output of results
    # print("Tweet Id\t\tGround Truth\tPredicted")
    for tweet_id in y_test:
        labels[tweet_id] = "correct" if y_test[tweet_id] == classifications[tweet_id] else "wrong"
        # print(f"{tweet_id}\t| {y[tweet_id]}\t\t| {classifications[tweet_id]}\t|{labels[tweet_id]}")


    """
    Data for Trace File: 
        1. TweetId can be found in tweets.keys()
        2. Most likely class can be found in classifications[tweet_id]
        3. Score of most likely class can be found in score[tweet_id]
        4. Ground truth can be gound in y[tweet_id]
        5. Label can be found in labels[tweet_id]
    """

    #UNCOMMENT FOR ORIGINAL VOCABULARY
    # out('OV', x_test, classifications, score, y_test, labels) 
    #UNCOMMENT FOR FILTERED VOCABULARY
    # out('FV', x_test, classifications, score, y_test, labels)