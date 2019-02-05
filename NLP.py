# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 05:29:10 2019

@author: SHAJI JAMES
"""

import tweepy
consumer_key = '8vYFAGp9JL6pu13n1Dd1HBKw2'
consumer_secret = '2vWm3WCR3fgL02qK5XtibbrLJvpKHRXcAVHZyu61cGtTzWdeU7'
access_token = '2576378914-V3wsVr0cx3x8or7tMtI7KVbbYrD7Y1OmW8uE3aw'
access_token_secret = 'TKyymNul0qYLYexzvEkx2BzohaSmefaq1GDj7eMfuVrjb'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
    
fetched_tweets = api.search(q = '#digitalindia', count = 100,lang='en')

tweets=[]
for tweet in fetched_tweets: 
    if tweet.retweet_count > 0: 
        if tweet.text not in tweets: 
            tweets.append(tweet.text) 
    else: 
        tweets.append(tweet.text)

len(tweets)

import re
def clean_tweet(tweet):
    return str.lower(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", ' ', tweet))

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

refined_tweets=[]
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
for tweet in tweets:
    tokens=word_tokenize(clean_tweet(tweet))
    word_list=[]
    for word in tokens:
        if word not in stop_words:
            word_list.append(lemmatizer.lemmatize(word, pos='v'))
    refined_tweet=' '.join(word_list)
    refined_tweets.append(refined_tweet)

# =============================================================================
# #sentiment analysis
# =============================================================================

polarity=0
positive=0
negative=0
neutral=0

from textblob import TextBlob
for tweet in refined_tweets:
    blob=TextBlob(tweet)
    polarity+=blob.sentiment.polarity
    if blob.sentiment.polarity>0:
        positive+=1
    elif blob.sentiment.polarity==0:
        neutral+=1
    else:
        negative+=1

print('Positive tweet percentage: ',(positive/len(tweets))*100)
print('Negative tweet percentage: ',(negative/len(tweets))*100)
print('Neutral tweet percentage: ',(neutral/len(tweets))*100)


# =============================================================================
# #topic modeling
# =============================================================================
refined_tokens=[]
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
for tweet in tweets:
    tokens=word_tokenize(clean_tweet(tweet))
    word_list=[]
    for word in tokens:
        if word not in stop_words:
            word_list.append(lemmatizer.lemmatize(word, pos='v'))
    refined_tokens.append(word_list)

import gensim
from gensim import corpora
dictionary = corpora.Dictionary(refined_tokens)

#visualising dictionary
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

#Preparing document term matrix
doc_term_matrix = [dictionary.doc2bow(token) for token in refined_tokens]

#visualising the occurence of terms in a single document
doc_20=doc_term_matrix[20]
for i in range(len(doc_20)):
    print('No: ',doc_20[i][0],'Word: ',dictionary[doc_20[i][0]],'| Occurence(s): ',doc_20[i][1])

#transformed corpus
tfidf = gensim.models.TfidfModel(doc_term_matrix)
corpus_tfidf = tfidf[doc_term_matrix]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

#Running LDA using Bag of Words
ldamodel = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)
ldamodel.print_topics(num_topics=10, num_words=3)

#Running LDA using TF-IDF
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=50)
lda_model_tfidf.print_topics(num_topics=10, num_words=3)

#evaluating lda bag of words model for a single document
for index, score in ldamodel[doc_term_matrix[20]]:
    print('Score: ',score,'Topic:', ldamodel.print_topic(index, 3))

#evaluating lda tf-idf model for a single document   
for index, score in lda_model_tfidf[doc_term_matrix[20]]:
    print('Score: ',score,'Topic:', lda_model_tfidf.print_topic(index, 3))

unseen_document = 'Digital India is a great initiative to begin with. If executed properly, a great rise in the development of India can be seen soon enough.'
test_token=word_tokenize(clean_tweet(unseen_document))
test_word_list=[]
for word in test_token:
    if word not in stop_words:
        test_word_list.append(lemmatizer.lemmatize(word, pos='v'))
test_dictionary = corpora.Dictionary([test_word_list])
bow_vector = test_dictionary.doc2bow(test_word_list)

for index, score in ldamodel[bow_vector]:
    print('Score: ',score,'Topic:', ldamodel.print_topic(index, 3))

# =============================================================================
# #extractive summarization
# =============================================================================
tweet_list=[]
for tweet in tweets:
    tweet_list.append(clean_tweet(tweet))
    
tweets_doc='. '.join(tweet_list)

from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

print(summarize(tweets_doc, ratio=0.5))
print(keywords(tweets_doc, ratio=0.01))