import csv
import string
from nltk.corpus import stopwords

if __name__ == '__main__':
    tsv_file = open("covid_training.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    next(read_tsv) #ignore header


    list_of_tweets = []
    for row in read_tsv:
        list_of_tweets.append(row[1].lower())

    tokenized_tweets = []
    for tweet in list_of_tweets:
        temp = [char for char in tweet if char not in string.punctuation]
        temp = ''.join(temp)
        tokenized_tweets.append(temp.split())
    print(tokenized_tweets)


