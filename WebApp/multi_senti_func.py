
# coding: utf-8
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

import matplotlib.pyplot as plt
import goslate
#import urllib.request
from googletrans import Translator
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment import util 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

debug = False
test = True

#translation
def translation_to_eng(text):
    translator = Translator()
    trans = translator.translate(text, 'en').text
    return trans


# tokenization
def tokenization(text):
    # tokenize the sentence and word
    # text
    sent_token = nltk.sent_tokenize(text)
    wtokens = []
    for sent in sent_token:
        if sent is not None:
            tok = nltk.word_tokenize(str(sent))
            wtokens.append(tok)
        else:
            wtokens.append(None)
    return wtokens


# generate POS tag
def pos_mark(wtokens):
    pos_tag=[]
    for tk in wtokens:        
         pos_tag.append(nltk.pos_tag(tk))        
    return pos_tag


#gainning each words' sentiment score in each sentence
def senti_score(pos_tag):
    nltk.download('sentiwordnet')
    wnl = nltk.WordNetLemmatizer()
    score_list=[] 
    last_lemma = 'aa'
    for idx,taggedsent in enumerate(pos_tag): # loop all the sentence in POS tag
        score_list.append([])
        for idx2,t in enumerate(taggedsent): #loop all the word in each sentence POS tag
            newtag=''
            lemmatized=wnl.lemmatize(t[0].lower()) # t[0]: original tokened word, and change it to LEMMA
            # transfer Penn Treebank POS to sentiwordnet POS tag
            if t[1].startswith('NN'): #t[1]: each POS of words
                newtag='n' #Noun
            elif t[1].startswith('JJ'):
                newtag='a' #Adjective
            elif t[1].startswith('V'):
                newtag='v' #Verb
            elif t[1].startswith('R'):
                newtag='r' #Adverb
            else:
                newtag=''       
            if(newtag!=''):    
                synsets = list(swn.senti_synsets(lemmatized, newtag)) # for each word there is a list of synonyms
                #count all synonyms avg sentiment score as the sentiment score for this word       
                score=0 
                if(len(synsets)>0):
                    for syn in synsets: 
                        score+=syn.pos_score()-syn.neg_score() # add them to total score
                        
                    if lemmatized == 'not' or lemmatized == 'no' or lemmatized == 'Not' or lemmatized == 'No' or lemmatized == 'Too' or lemmatized == 'too':
                        score_list[idx].append(0)
                    else:
                        if last_lemma == 'not' or last_lemma == 'no' or lemmatized == 'Not' or lemmatized == 'No' or lemmatized == 'Too' or lemmatized == 'too':
                            score_list[idx].append(-score/len(synsets))
                            #print(lemmatized)
                            #print(-score/len(synsets))
                        else:
                            score_list[idx].append(score/len(synsets))
                    last_lemma = lemmatized
                        
                           

    #gaining each sentence sentiment score
    sentence_sentiment=[]
    for score_sent in score_list:
        if len(score_sent) > 0:
            sentence_sentiment.append(sum([word_score for word_score in score_sent]))
        else:
            sentence_sentiment.append(float(0))
    sentence_senti_score = sum(sentence_sentiment)
    return sentence_senti_score


def vader_senti_score(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    com_score = vs['compound']
    return com_score
    

def adjusted_score(s1,s2,s3,s4):
    adj_score = s1+2*s2+s3+s4
    return adj_score
