# Implementation of RAKE - Rapid Automatic Keyword Extraction algorithm
# as described in:
# Rose, S., D. Engel, N. Cramer, and W. Cowley (2010). 
# Automatic keyword extraction from individual documents.
# In M. W. Berry and J. Kogan (Eds.), Text Mining: Applications and Theory.unknown: John Wiley and Sons, Ltd.
#
# NOTE: The original code (from https://github.com/aneesha/RAKE)
# has been extended by a_medelyan (zelandiya)
# with a set of heuristics to decide whether a phrase is an acceptable candidate
# as well as the ability to set frequency and phrase length parameters
# important when dealing with longer documents
#
# NOTE 2: The code published by a_medelyan (https://github.com/zelandiya/RAKE-tutorial)
# has been additionally extended by Marco Pegoraro to implement the adjoined candidate
# feature described in section 1.2.3 of the original paper. Note that this creates the
# need to modify the metric for the candidate score, because the adjoined candidates
# have a very high score (because of the nature of the original score metric)

from __future__ import absolute_import
from __future__ import print_function

import json
import operator
import re
from collections import Counter

import six
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from six.moves import range
import numpy as np
import pandas as pd
# from nltk.parse.stanford import StanfordDependencyParser
from nltk.corpus import wordnet
debug = False
test = True

# path_to_jar = '/Users/yutingan/stanford-parser-full-2017-06-09/stanford-parser.jar'
# path_to_models_jar = '/Users/yutingan/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar'
# dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words


def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words


def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[\\[\\]\n.!?,;:\t\\-\\"\\(\\)\\\'\u2019\u2013]')
    sentences = sentence_delimiters.split(text)
    return sentences


def build_stop_word_regex(stop_word_list):
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = '\\b' + word + '\\b'
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern


#
# Function that extracts the adjoined candidates from a list of sentences and filters them by frequency
#
def extract_adjoined_candidates(sentence_list, stoplist, min_keywords, max_keywords, min_freq):
    adjoined_candidates = []
    for s in sentence_list:
        # Extracts the candidates from each single sentence and adds them to the list
        adjoined_candidates += adjoined_candidates_from_sentence(s, stoplist, min_keywords, max_keywords)
    # Filters the candidates and returns them
    return filter_adjoined_candidates(adjoined_candidates, min_freq)


# return adjoined_candidates

#
# Function that extracts the adjoined candidates from a single sentence
#
def adjoined_candidates_from_sentence(s, stoplist, min_keywords, max_keywords):
    # Initializes the candidate list to empty
    candidates = []
    # Splits the sentence to get a list of lowercase words
    sl = s.lower().split()
    # For each possible length of the adjoined candidate
    for num_keywords in range(min_keywords, max_keywords + 1):
        # Until the third-last word
        for i in range(0, len(sl) - num_keywords):
            # Position i marks the first word of the candidate. Proceeds only if it's not a stopword
            if sl[i] not in stoplist:
                candidate = sl[i]
                # Initializes j (the pointer to the next word) to 1
                j = 1
                # Initializes the word counter. This counts the non-stopwords words in the candidate
                keyword_counter = 1
                contains_stopword = False
                # Until the word count reaches the maximum number of keywords or the end is reached
                while keyword_counter < num_keywords and i + j < len(sl):
                    # Adds the next word to the candidate
                    candidate = candidate + ' ' + sl[i + j]
                    # If it's not a stopword, increase the word counter. If it is, turn on the flag
                    if sl[i + j] not in stoplist:
                        keyword_counter += 1
                    else:
                        contains_stopword = True
                    # Next position
                    j += 1
                # Adds the candidate to the list only if:
                # 1) it contains at least a stopword (if it doesn't it's already been considered)
                # AND
                # 2) the last word is not a stopword
                # AND
                # 3) the adjoined candidate keyphrase contains exactly the correct number of keywords (to avoid doubles)
                if contains_stopword and candidate.split()[-1] not in stoplist and keyword_counter == num_keywords:
                    candidates.append(candidate)
    return candidates


#
# Function that filters the adjoined candidates to keep only those that appears with a certain frequency
#
def filter_adjoined_candidates(candidates, min_freq):
    # Creates a dictionary where the key is the candidate and the value is the frequency of the candidate
    candidates_freq = Counter(candidates)
    filtered_candidates = []
    # Uses the dictionary to filter the candidates
    for candidate in candidates:
        freq = candidates_freq[candidate]
        if freq >= min_freq:
            filtered_candidates.append(candidate)
    return filtered_candidates


def generate_candidate_keywords(sentence_list, stopword_pattern, stop_word_list, min_char_length=1, min_words_length=2, max_words_length=5,
                                min_words_length_adj=1, max_words_length_adj=1, min_phrase_freq_adj=5):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "" and is_acceptable(phrase, min_char_length,min_words_length, max_words_length):
                phrase_list.append(phrase)
    phrase_list += extract_adjoined_candidates(sentence_list, stop_word_list, min_words_length_adj,
                                               max_words_length_adj, min_phrase_freq_adj)
    return phrase_list
### generate lemmatized a sentence
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.name() != word:
                synonyms.append(l.name())
            if l.antonyms():
                synonyms.append('no_' + l.antonyms()[0].name())
    return set(synonyms)


def lemmatize_sentence(s):
    verb_list=['is','are','am']
    lemma=WordNetLemmatizer()
    new_sentence=""
    word_token=word_tokenize(s)
    pos_tagging=nltk.pos_tag(word_token)
    for pair in pos_tagging:
        word_net_pos=get_wordnet_pos(pair[1])
        if word_net_pos!='' and pair[0] not in verb_list:
            word=lemma.lemmatize(pair[0],get_wordnet_pos(pair[1]))
        else:
            word=pair[0]
        if new_sentence=="":
            new_sentence+=word
        else:
            new_sentence+=" "+word
    return new_sentence

def lemmatize_sentence_list(sentence_list):
    new_list=[]
    for s in sentence_list:
        new_s=lemmatize_sentence(s)
        new_list.append(new_s)
    return new_list


def is_acceptable(phrase, min_char_length, min_words_length,max_words_length):
    # a phrase must have a min length in characters
    if len(phrase) < min_char_length:
        return 0

    # a phrase must have a max number of words
    words = phrase.split()
    if len(words) > max_words_length or len(words)<min_words_length:
        return 0

    digits = 0
    alpha = 0
    for i in range(0, len(phrase)):
        if phrase[i].isdigit():
            digits += 1
        elif phrase[i].isalpha():
            alpha += 1

    # a phrase must have at least one alpha character
    if alpha == 0:
        return 0

    # a phrase must have more alpha than digits characters
    if digits > alpha:
        return 0
    return 1


def calculate_word_metrics(phrase_list, tradeoff):
    word_frequency = {}
    word_degree = {}
    word_score = {}
    keyphrase_counts = {}
    synonyms_set={}

    for phrase in phrase_list:
        keyphrase_counts.setdefault(phrase, 0)
        keyphrase_counts[phrase] += 1

        word_list = separate_words(phrase, 0)

        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1

            word_degree.setdefault(word, 0)
            word_degree[word] += len(word_list) - 1

    for word in word_frequency.keys():
        synonyms=get_synonyms(word)
        synonyms_set[word]=synonyms
        for s in synonyms:
            if s in word_frequency.keys():
                word_frequency[word]+=word_frequency[s]

    for word in word_frequency.keys():
        word_score.setdefault(word, 0)
        if word_degree[word] != 0:
            word_score[word] = word_frequency[word] / np.power(word_degree[word], tradeoff)
        else:
            word_score[word] = word_frequency[word]

    keyphrase_freq = {kp: float(sc) / len(phrase_list) for (kp, sc) in keyphrase_counts.items()}

    return word_score, keyphrase_counts, keyphrase_freq, synonyms_set

def calculate_stem_metrics(phrase_list,tradeoff):
    word_count = {}
    word_degree = {}
    word_score = {}
    word_frequency={}

    for phrase in phrase_list:

        word_list = separate_words(phrase, 0)

        for word in word_list:
            word_count.setdefault(word, 0)
            word_count[word] += 1

            word_degree.setdefault(word, 0)
            word_degree[word] += len(word_list) - 1

    for word in word_count.keys():
        word_score.setdefault(word, 0)
        if word_degree[word] != 0:
            word_score[word] = word_count[word] / np.power(word_degree[word], tradeoff)
        else:
            word_score[word] = word_count[word]

    word_frequency = {kp: float(sc) / len(word_count) for (kp, sc) in word_count.items()}

    return word_score, word_count, word_frequency


def generate_candidate_keyphrase_scores(final_list, word_scores, keyphrase_stem_count, track_stem):
    keyphrase_candidates_score = {}
    for phrase in final_list:
        phrase_stem = track_stem[phrase]
        keyphrase_candidates_score.setdefault(phrase, 0)
        word_list = separate_words(phrase_stem, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_scores[word]
        keyphrase_candidates_score[phrase] = float(candidate_score) / len(word_list)

    keyphrase_candidates_count = {kp: keyphrase_stem_count[track_stem[kp]] for kp in
                            keyphrase_candidates_score.keys()}

    return keyphrase_candidates_score, keyphrase_candidates_count

def generate_lemma_keyphrase_scores(phrase_lemma_list,word_scores):
    keyphrase_candidates_score={}
    for phrase in phrase_lemma_list:
        word_list=separate_words(phrase,0)
        candidate_score = 0
        for word in word_list:
            candidate_score+=word_scores[word]
        keyphrase_candidates_score[phrase]=float(candidate_score)/len(word_list)
    return keyphrase_candidates_score

###new implementation from last year
def spell_check(sentence_list):
    with open('dico_spell.json', 'r') as f:
        dico_spell = json.load(f)

    for i in range(len(sentence_list)):
        sentence_split = sentence_list[i].split(' ')
        for k in range(len(sentence_split)):
            if sentence_split[k] in dico_spell.keys():
                sentence_split[k] = dico_spell[sentence_split[k]]
        sentence_list[i] = str(' '.join(sentence_split))

    return sentence_list

###The old handle negation function, can not detect "no itchness, pale or wrinkles"

def handle_neg(candidate):
    candidate = candidate.lower()
    neg_items = ['not a lot of', 'not', 'no', 'non', 'not', 'nor', 'free of', 'not too', 'not to', 'clear of']
    for neg_item in neg_items:
        candidate = candidate.replace(neg_item, 'no')

    word_list = candidate.split(' ')
    if '' in word_list:
        word_list.remove('')

    to_remove = []
    new_phrases = []
    cpt = 0

    while cpt < len(word_list):

        if word_list[cpt][-4:] == 'less':
            new_phrases.append('no_' + word_list[cpt][:-4])

            to_remove += [word_list[cpt]]
            cpt += 1
            continue

        if cpt + 1 < len(word_list):
            if word_list[cpt + 1] == 'free':
                new_phrases.append('no_' + word_list[cpt])

                to_remove += [word_list[cpt], word_list[cpt + 1]]
                cpt += 2
                continue

        if word_list[cpt] == 'no' and cpt + 2 < len(word_list):
            if cpt+4<len(word_list):
                if word_list[cpt+3]=='or':
                    new_phrases.append('no_'+word_list[cpt+1])
                    new_phrases.append('no_'+word_list[cpt+2])
                    new_phrases.append('no_'+word_list[cpt+4])

                    to_remove += [word_list[cpt], word_list[cpt + 1], word_list[cpt + 2], word_list[cpt+3],word_list[cpt + 4]]
                    cpt += 4
                    continue

            if word_list[cpt + 2] == 'or' and cpt + 4 < len(word_list):
                if word_list[cpt + 4] != 'or':
                    new_phrases.append('no_' + word_list[cpt + 1])
                    new_phrases.append('no_' + word_list[cpt + 3])

                    to_remove += [word_list[cpt], word_list[cpt + 1], word_list[cpt + 2], word_list[cpt + 3]]
                    cpt += 4
                    continue

                else:
                    new_phrases.append('no_' + word_list[cpt + 1])
                    new_phrases.append('no_' + word_list[cpt + 3])
                    new_phrases.append('no_' + word_list[cpt + 5])

                    to_remove += [word_list[cpt], word_list[cpt + 1], word_list[cpt + 2], word_list[cpt + 3],
                                  word_list[cpt + 4], word_list[cpt + 5]]
                    cpt += 6
                    continue

            elif word_list[cpt + 2] == 'or' and cpt + 4 >= len(word_list):
                new_phrases.append('no_' + word_list[cpt + 1])
                new_phrases.append('no_' + word_list[cpt + 3])

                to_remove += [word_list[cpt], word_list[cpt + 1], word_list[cpt + 2], word_list[cpt + 3]]
                cpt += 4
                continue

            else:
                new_phrases.append('no_' + word_list[cpt + 1])

                to_remove += [word_list[cpt], word_list[cpt + 1]]
                cpt += 2
                continue

        elif word_list[cpt] == 'no' and cpt + 1 < len(word_list):
            new_phrases.append('no_' + word_list[cpt + 1])

            to_remove += [word_list[cpt], word_list[cpt + 1]]
            cpt += 2
            continue

        else:
            cpt += 1
            continue

    to_keep = [el for el in word_list if el not in to_remove]
    new_candidate = to_keep + new_phrases

    return ' '.join(new_candidate)


###newer version of handle negation. more accurate detect negation words.
'''
def handle_neg(sentence):
    print("Original Candidate: ",sentence)
    sentence = sentence.lower()
    sentence=sentence.replace(',',' ').replace('.',' ').replace('?',' ')
    neg_items = ['not a lot of', 'not', 'no', 'non', 'not', 'nor', 'free of', 'not too', 'not to', 'clear of']
    for neg_item in neg_items:
        sentence = sentence.replace(neg_item, 'no')


    result = dependency_parser.raw_parse(sentence)
    dep = result.__next__()
    dependency = []
    word_to_negate = ['no']
    new_phrases=[]
    for row in list(dep.triples()):
        relation = [row[0][0], row[0][1], row[1], row[2][0], row[2][1]]
        dependency.append(relation)
    df = pd.DataFrame(dependency, columns=['first_word', 'first_tag', 'relation', 'second_word', 'second_tag'])
    if 'neg' in list(df['relation']):
        after_no = df[df['second_word'] == 'no']['first_word']
        for word in list(after_no):
            word_to_negate.append(word)
            new_phrases.append('no_' + word)
            if not df[df['first_word'] == word][df['second_word'] == 'or'][df['relation'] == 'cc'].empty:
                for relation_word in list(df[df['first_word'] == word][df['relation'] == 'conj']['second_word']):
                    word_to_negate.append(relation_word)
                    new_phrases.append('no_' + relation_word)

    word_list = sentence.split(' ')
    while '' in word_list:
        word_list.remove('')
    for word in word_list:
        if word_list[-4:]=='less':
            word_to_negate.append(word)
            new_phrases.append('no_'+word[:-4])

    to_keep = [el for el in word_list if el not in word_to_negate]
    new_candidate = to_keep + new_phrases
    print("After Negation: ",' '.join(new_candidate))
    return ' '.join(new_candidate)

'''
def handle_neg_list(sentence_list):
    sent_handle_neg = []
    for candidate in sentence_list:
        print(candidate)
        if candidate and candidate not in [',',' ','.','!']:
            sent_handle_neg.append(handle_neg(candidate))
        else:
            continue
    return sent_handle_neg

'''
def group_words(sentence_list):
    return_list=[]
    for sentence in sentence_list:
        sentence_stream = sentence.split()
        one_sentence=bigram[sentence_stream]
        reformed_sentence=" ".join(one_sentence)
        return_list.append(reformed_sentence)
    return return_list
'''


def stem_candidate_keywords(phrase_list):
    stemmer = PorterStemmer()

    # Stem phrase_list and track
    phrase_list_stem = []
    track_stem = {}
    for phrase in phrase_list:
        spl = phrase.split(' ')
        phrase_stem = ''
        for item in spl:
            if len(phrase_stem) == 0:
                phrase_stem += stemmer.stem(item)

            else:
                phrase_stem += ' ' + stemmer.stem(item)

        phrase_list_stem.append(str(phrase_stem))
        track_stem[phrase] = str(phrase_stem)

    # Compute reverse tracking
    track_stem_rev = {}
    for phrase, stem_phrase in track_stem.items():
        if stem_phrase not in track_stem_rev.keys():
            track_stem_rev[stem_phrase] = [phrase]
        else:
            track_stem_rev[stem_phrase].append(phrase)

    # Compute final list of keyphrase candidates
    final_list = list(set(phrase_list_stem))
    final_list = [track_stem_rev[str(phrase)][0] for phrase in final_list]

    return final_list, phrase_list_stem, track_stem


# justify whether two phrases are similar. Two phrases are similar if they defer by only one words(if the phrase is more than three words)
# or all the same (if the phrase is less or equal than two words)
def similar_keyphrases(phrase1,phrase2, synonyms_set):
    similar=False
    word_list_1 = separate_words(phrase1, 0)
    word_list_2 = separate_words(phrase2, 0)
    same_count=0
    if len(word_list_1)>=len(word_list_2):
        for word in word_list_2:
            same=False
            synonyms=synonyms_set[word]
            synonyms.add(word)
            for syns in synonyms:
                if syns in word_list_1:
                    same=True
            if same==True:
                same_count+=1
        if len(word_list_2)>2 and same_count>=len(word_list_2)-1:
            similar = True
        if len(word_list_2)<=2 and same_count>len(word_list_2)-1:
            similar=True
    else:
        for word in word_list_1:
            same=False
            synonyms=synonyms_set[word]
            synonyms.add(word)
            for syns in synonyms:
                if syns in word_list_1:
                    same=True
            if same == True:
                same_count+=1
        if len(word_list_1)>2 and same_count>=len(word_list_1)-1:
            similar = True
        if len(word_list_1)<=2 and same_count>len(word_list_1)-1:
            similar=True
    return similar

def remove_similar_keyphrases(keyphrases, synonyms_set):
    i=0
    while i<len(keyphrases):
        gold_phrase=keyphrases[i]
        for rest_phrase in keyphrases[i:]:
            similar=similar_keyphrases(gold_phrase[0],rest_phrase[0],synonyms_set)
            if similar == True:
                print(gold_phrase, " and ",rest_phrase, " are similar phrases")
                keyphrases.remove(rest_phrase)
        i=i+1
    return keyphrases

class Rake(object):
    def __init__(self, stop_words_path, min_char_length=2, min_words_length=2,max_words_length=7, min_keyword_frequency=1,
                 min_words_length_adj=1, max_words_length_adj=3, min_phrase_freq_adj=2):
        self.__stop_words_path = stop_words_path
        self.__stop_words_list = load_stop_words(stop_words_path)
        self.__min_char_length = min_char_length
        self.__max_words_length = max_words_length
        self.__min_keyword_frequency = min_keyword_frequency
        self.__min_words_length_adj = min_words_length_adj
        self.__max_words_length_adj = max_words_length_adj
        self.__min_phrase_freq_adj = min_phrase_freq_adj
        self.__min_words_length=min_words_length

    def run(self, text,freq_trade_off,top_n):
        sentence_list_raw = split_sentences(text)
        sentence_list=spell_check(sentence_list_raw)

        stop_words_pattern = build_stop_word_regex(self.__stop_words_list)
        sentence_list_neg_melt = handle_neg_list(sentence_list)
        sentence_list_lemma=lemmatize_sentence_list(sentence_list_neg_melt)

        ## new step for extracting common word phrases like body wash
        #sentence_list_group_words=group_words(sentence_list_neg_melt)
        phrase_list_raw = generate_candidate_keywords(sentence_list_lemma, stop_words_pattern, self.__stop_words_list,
                                                  self.__min_char_length,self.__min_words_length, self.__max_words_length,
                                                  self.__min_words_length_adj, self.__max_words_length_adj,
                                                  self.__min_phrase_freq_adj)

        final_list, phrase_list_stem, track_stem = stem_candidate_keywords(phrase_list_raw)
        stem_word_scores,keyphrase_stem_counts,keyphrase_stem_frequency = calculate_stem_metrics(phrase_list_stem, freq_trade_off)
        lemma_word_scores,keyphrase_lemma_counts,keyphrase_lemma_frequency,synonyms_set=calculate_word_metrics(phrase_list_raw,freq_trade_off)
        keyphrase_candidates_score = generate_lemma_keyphrase_scores(phrase_list_raw, lemma_word_scores)


        return_list={}
        sort_keywords = sorted(six.iteritems(keyphrase_candidates_score), key=operator.itemgetter(1), reverse=True)
        sorted_keywords = remove_similar_keyphrases(sort_keywords,synonyms_set)
        #top_n=int(len(sorted_keywords)/3)
        if top_n>len(sorted_keywords):
            top_n_keywords=sorted_keywords
        else:
            top_n_keywords=sorted_keywords[0:top_n]
        keywords_score=[]
        keywords_counts=[]
        keywords_freq=[]
        for pair in top_n_keywords:
            keywords_score.append((pair[0],pair[1]))
            keywords_counts.append((pair[0],keyphrase_lemma_counts[pair[0]]))
            keywords_freq.append([pair[0],keyphrase_lemma_frequency[pair[0]]])

        sorted_stem = sorted(six.iteritems(stem_word_scores), key=operator.itemgetter(1), reverse=True)
        #n_stem=int(len(sorted_stem)/3)
        if top_n>len(sorted_stem):
            top_n_stem=sorted_stem
        else:
            top_n_stem = sorted_stem[0:top_n]
        stem_score = []
        stem_counts = []
        stem_freq = []
        for pair in top_n_stem:
            stem_score.append((pair[0], pair[1]))
            stem_counts.append((pair[0], keyphrase_stem_counts[pair[0]]))
            stem_freq.append([pair[0], keyphrase_stem_frequency[pair[0]]])
        return_list['keywords_score']=keywords_score
        return_list['keywords_counts']=keywords_counts
        return_list['keywords_freq']=keywords_freq
        return_list['stem_score']=stem_score
        return_list['stem_counts']=stem_counts
        return_list['stem_freq']=stem_freq


        return keywords_score,keywords_counts,stem_counts


