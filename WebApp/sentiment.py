import pandas as pd
import numpy as np
import time
import WebApp.apis.aylienapi.aylienapiclient.textapi


def calculate_score(polarity, polarity_conf):
    if polarity == "neutral":
        return 3
    elif polarity == "negative":
        if polarity_conf > 0.5:
            return 1
        return 2
    else:
        if polarity_conf > 0.5:
            return 5
        return 4

# folder = "data/"
# df = pd.read_excel(folder + "Babo-Botanicals-Oatmilk-Calendula-Moisturizing_part2.xlsx")
# text = df['text'].tolist()
# summary = df['title'].tolist()
# rating = df['stars'].tolist()

# num_reviews = len(rating)


# AYLIEN
# aylien = apis.aylienapi.aylienapiclient.textapi.Client("4df6473c", "f827888e31b6b52f85a6061eb3f18ad1")
# aylien = apis.aylienapi.aylienapiclient.textapi.Client("8f14979e", "4b1ff15f606a003e025a93070a822d54")

# # Textalytics
# api = 'http://api.meaningcloud.com/sentiment-2.0'
# key = '6c46872fd5bded758034d0e5c6d1cf00'
# model = 'general_es' # general_es / general_es / general_fr

def get_sentiment(t, s):
    aylien = apis.aylienapi.aylienapiclient.textapi.Client("4969e38e", "f8de4ced275a6b449a677d3efeae6e5b")

    text_sentiment = aylien.Sentiment({'text': t})
#     print(text_sentiment)
    text_polarity = text_sentiment['polarity']
    text_polarity_conf = text_sentiment['polarity_confidence']

    text_score = calculate_score(text_polarity, text_polarity_conf)

    sum_sentiment = aylien.Sentiment({'text': s})
    sum_polarity = sum_sentiment['polarity']
    sum_polarity_conf = sum_sentiment['polarity_confidence']

    sum_score = calculate_score(sum_polarity, sum_polarity_conf)
    
    return text_score, sum_score
# AYLIEN
# text_matrix = np.zeros([5,5])
# sum_matrix = np.zeros([5,5])


# for file in files:
#     text_fd = open('text_score.out', 'w')
#     sum_fd = open('sum_score.out', 'w')

#     i = 0
#     while(i < num_reviews):
        
#         t = text[i]
#         s = summary[i]
#         r = int(rating[i])

#         text_sentiment = aylien.Sentiment({'text': t})
#     #     print(text_sentiment)
#         text_polarity = text_sentiment['polarity']
#         text_polarity_conf = text_sentiment['polarity_confidence']

#         text_score = calculate_score(text_polarity, text_polarity_conf, r)
#         text_matrix[r-1][text_score-1] += 1

#         sum_sentiment = aylien.Sentiment({'text': s})
#         sum_polarity = sum_sentiment['polarity']
#         sum_polarity_conf = sum_sentiment['polarity_confidence']

#         sum_score = calculate_score(sum_polarity, sum_polarity_conf, r)
#         sum_matrix[r-1][sum_score-1] += 1
        
#         print(str(text_score) + "," + str(sum_score) + "," + str(r))
#         text_fd.write(str(text_score) + "," + str(r))
#         sum_fd.write(str(sum_score) + "," + str(r))

#         i += 1
#         if i%25 == 0:
#             time.sleep(120)

        
        
#     text_fd.close()
#     sum_fd.close()



