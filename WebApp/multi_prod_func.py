import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

import matplotlib.pyplot as plt
from googletrans import Translator
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment import util 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

debug = False
test = True

#translation
def translation_to_eng(df):
    emoji_pattern = re.compile("[" # remove emoji from the review text
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    translator = Translator()
    text_trans = []
    for i in range(len(df)):
        text = df[i]
        text_cl = str(emoji_pattern.sub(r'', text))
        if text_cl is not None:
            try:
                trans = translator.translate(text_cl, 'en')
                text_trans.append(trans.text)
            except:
                print("*****")
                print(i)
                print(text_cl)
        else:
            text_trans.append(None)
    return text_trans

# tokenization
def tokenization(df):
    nltk.download('sentiwordnet')
    reviews = [nltk.sent_tokenize(lines) for lines in df]
    review_tokens = []
    #neg_list = [] # all negetion dependent word
    for rev in reviews:
        word_token = []
        review_neg_word = [] #every review neg dependent word
        for sent in rev:
            if sent is not None:
                ##detect negation
                #result = dependency_parser.raw_parse(str(sent))
                #dep = result.__next__()
                #parsing_list = list(dep.triples())
                #sent_neg_word = [] #every sentence neg dependent word
                #for row in parsing_list:
                #    if row[1] == 'neg':
                #        word = row[0][0]
                #        sent_neg_word.append(word)
                #review_neg_word.append(sent_neg_word)
                
                ##append tokenization

                tok = nltk.word_tokenize(str(sent))
                word_token.append(tok)
            else:
                #b_neg.append(None)
                word_token.append(None)
        #neg_list.append(review_neg_word)
        flattened_token = sum(word_token, [])
        review_tokens.append(flattened_token)
    return review_tokens

# POS
def pos_mark(wtokens):
    pos_tag=[]
    for tk in wtokens:        
         pos_tag.append(nltk.pos_tag(tk))        
    return pos_tag

#Senti Score
def senti_score(pos_tag):
    wnl = nltk.WordNetLemmatizer()
    score_list=[] 
    last_lemma = 'aa'
    for idx,taggedsent in enumerate(pos_tag): # loop all the reviews in POS tag
        score_list.append([])
        for idx2,t in enumerate(taggedsent): #loop all the word in each reviews POS tag
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
                        
                    if lemmatized == 'not' or lemmatized == 'no' or lemmatized == 'without':
                        score_list[idx].append(0)
                    else:
                        if last_lemma == 'not' or last_lemma == 'no' or lemmatized == 'without':
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
    #print("Sentiment for each sentence for:"+doc)
    #print(sentence_sentiment)
    return sentence_sentiment

#vader
def vader_senti_score(df):
    analyzer = SentimentIntensityAnalyzer()
    vader_text_score = []
    for sentence in df:
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(sentence)
        com_score = vs['compound']
        vader_text_score.append(com_score)
    return vader_text_score

def adj_score_avg(df):
	df['adj_score'] = df['text_Sentiment_Score'] + 2 * df['title_Sentiment_Score']
	df['adj_senti_vader_score'] = df['Vader_text_score'] + df['Vader_title_score'] + df['adj_score']
	product_score = df['adj_senti_vader_score'].mean()
	return product_score


