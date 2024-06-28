from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow.pyfunc
from nltk.stem import PorterStemmer
import re
import pickle
import os


class NBmodel(mlflow.pyfunc.PythonModel):
    logprior = 0
    loglikelihood = {}
    stop_words = []


    def __init__(self):
        if os.path.exists("resources/stopwords.pkl"):
            with open("resources/stopwords.pkl", 'rb') as f:
                self.stop_words = pickle.load(f)
        else:
            nltk.download('stopwords')
    
    
    def load_context(self, context):
        # Load any artifacts required for the model
        with open(context.artifacts["logprior"], "rb") as f:
            self.logprior = pickle.load(f)
        
        with open(context.artifacts["loglikelihood"], "rb") as f:
            self.loglikelihood = pickle.load(f)

    def predict(self, context, model_input):
        model_input = model_input['text']
        result = self.classify_tweets(model_input, self.logprior, self.loglikelihood)
        return result
    

    def fit(self, X_train, y_train):
        X_train = X_train['text']
        freqs = self.count_tweets({}, X_train, y_train)
        self.logprior, self.loglikelihood = self.train_naive_bayes(freqs, y_train)
        print("Training successful")

#-----------------------------------------------------------

    def score(self,X_test, y_test=None):
        return self.test_naive_bayes(X_test, y_test, self.logprior, self.loglikelihood)

    def predict_single(self, tweet):
        return self.naive_bayes_predict(tweet, self.logprior, self.loglikelihood)
    

    def count_tweets(self, result, tweets, ys):
        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')

        for y, tweet in zip(ys, tweets):
            tweet = re.sub(r'\$\w*', '', tweet)  # remove stock market tickers like $GE
            tweet = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"
            tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)  # remove hyperlinks
            tweet = re.sub(r'#', '', tweet)  # remove hashtags

            tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
            tweet_tokens = tokenizer.tokenize(tweet)

            for word in tweet_tokens:
                if (word not in stopwords_english and  # remove stopwords
                        word not in string.punctuation):  # remove punctuation
                    stem_word = stemmer.stem(word)  # stemming word

                    pair = (stem_word, y)

                    if pair in result:
                        result[pair] += 1
                    else:
                        result[pair] = 1

        return result

    def train_naive_bayes(self, freqs, y_train):
        loglikelihood = {}
        logprior = 0

        vocab = set([pair[0] for pair in freqs.keys()])  # the first element of pair is the word
        V = len(vocab)  # number of distinct vocab(word)

        N_pos = N_neg = 0
        for pair in freqs.keys():
            if pair[1] > 0:
                N_pos += freqs[pair]
            else:
                N_neg += freqs[pair]

        D = len(y_train)
        D_pos = sum(y_train)
        D_neg = D - D_pos

        logprior = np.log(D_pos) - np.log(D_neg)

        for word in vocab:
            freq_pos = self.lookup(freqs, word, 1)
            freq_neg = self.lookup(freqs, word, 0)

            p_w_pos = (freq_pos + 1) / (N_pos + V)
            p_w_neg = (freq_neg + 1) / (N_neg + V)

            loglikelihood[word] = np.log(p_w_pos / p_w_neg)

        return logprior, loglikelihood

    def naive_bayes_predict(self, tweet, logprior, loglikelihood):
        word_l = self.process_tweet(tweet)
        p = logprior

        for word in word_l:
            if word in loglikelihood:
                p += loglikelihood[word]

        return p

    def classify_tweets(self, tweets, logprior, loglikelihood):
        p_count = 0
        n_count = 0
        neu_count = 0

        for tweet in tweets:
            prob = self.naive_bayes_predict(tweet, logprior, loglikelihood)
            if prob <= 0:
                n_count += 1
            elif prob <= 1:
                neu_count += 1
            else:
                p_count += 1

        n = len(tweets)
        p_count /= n
        n_count /= n
        neu_count /= n

        return {"positive": p_count, "neutral": neu_count, "negative": n_count}

    def test_naive_bayes(self, test_x, test_y, logprior, loglikelihood):
        y_hats = []
        for tweet in test_x:
            if self.naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
                y_hat_i = 1
            else:
                y_hat_i = 0

            y_hats.append(y_hat_i)

        scores = self.cal_scores(test_y, y_hats)

        return scores

    def cal_scores(self, y_test, y_hats):
        accuracy = accuracy_score(y_test, y_hats)
        precision = precision_score(y_test, y_hats)
        recall = recall_score(y_test, y_hats)
        f1 = f1_score(y_test, y_hats)

        return {"accuracy": accuracy, "precision": precision, "recall":recall,"f1 score": f1}

    def process_tweet(self, tweet):
        stemmer = PorterStemmer()
        stopwords_english = stopwords.words('english')

        tweet = re.sub(r'\$\w*', '', tweet)  # remove stock market tickers like $GE
        tweet = re.sub(r'^RT[\s]+', '', tweet)  # remove old style retweet text "RT"
        tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)  # remove hyperlinks
        tweet = re.sub(r'#', '', tweet)  # remove hashtags
        tweet = re.sub(r'\@\w*', '', tweet)  # remove mention
        tweet = re.sub(r'\d+', '', tweet) # remove number

        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation
                stem_word = stemmer.stem(word)  # stemming word
                tweets_clean.append(stem_word)

        return tweets_clean
    

    def test_lookup(func):
        freqs = {('sad', 0): 4,
                ('happy', 1): 12,
                ('oppressed', 0): 7}
        word = 'happy'
        label = 1
        if func(freqs, word, label) == 12:
            return 'SUCCESS!!'
        return 'Failed Sanity Check!'


    def lookup(self, freqs, word, label):
        '''
        Input:
            freqs: a dictionary with the frequency of each pair (or tuple)
            word: the word to look up
            label: the label corresponding to the word
        Output:
            n: the number of times the word with its corresponding label appears.
        '''
        n = 0  # freqs.get((word, label), 0)

        pair = (word, label)
        if (pair in freqs):
            n = freqs[pair]

        return n
