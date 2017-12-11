import os
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import sys
#from importlib import reload
#reload(sys)

from nltk.parse.stanford import StanfordParser


DELIMITER = '\n' + '*' * 30 + ' '
# For adding periods to the ends of the sentences
eos_regex = r',?\s*([^.])$'

cwd = os.getcwd() + '/'
path_to_jar = cwd + 'StanfordCoreNLP/stanford-corenlp-3.2.0.jar'
path_to_models_jar = cwd + 'StanfordCoreNLP/stanford-corenlp-3.2.0-models.jar'

stan_parser = StanfordParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
#ENCHANT_DICT = enchant.Dict("en_US")

def make_sentences_from_dataframe(df, columns):
	"""
	Concatenate columns of data frame into list of lists of sentences
	:param df: pd.DataFrame
	:param columns: list of strings of columns to convert to documents
	:return: list of lists of concatenated sentences for each column, + column names for each sentence list
	"""
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	# Make columns strings
	df.columns = [str(col) for col in df.columns]
	# Strip whitespace from column names
	df.columns = [col.strip() for col in df.columns]

	if len(columns) == 0:
		# Just use all string columns
		columns = [col.strip() for col in df.columns if str(df[col].dtype) == 'object']

	sentence_sets = []
	for col in columns:
		# df[col] = df[col].str.replace(eos_regex, r'\1.')
		text_blob = df[col].str.cat(sep=' ').encode('utf-8').strip()
		tokenized = tokenizer.tokenize(text_blob)
		sentence_sets.append(tokenized)

	#print(sentence_sets[0][:2])
	return np.array(sentence_sets), columns


def make_sentences_by_group(df, group_by_col, column):
	"""
	Concatenate columns of dataframe into list of sentences, by group.
	:param df: pd.DataFrame
	:param column: String - column which contains text to concatenate for each group
	:param group_by_col: String - column on which to group
	:return: List<List<String>> - List of lists of sentences
	"""
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	sentence_sets = []
	unique_groups = df[group_by_col].unique()
	for group in unique_groups:
		df[column] = df[column].str.replace(eos_regex, r'\1.')
		sentences = df[df[group_by_col] == group][column].str.cat(sep=' ')
		tokenized = tokenizer.tokenize(sentences)
		sentence_sets.append(tokenized)

	#print(sentence_sets[0][:2])
	return sentence_sets, unique_groups


def split_long_sentences(sentences, l):
	"""
	Split longer sentences so that they don't always take precedence in vector distance computation.
	Todo - worthy of optimization
	:param: sentences - list of sentences
	:param: l - if sentence word count is longer than l, split into phrases with length l
	:return: List of split sentences, which will be longer than the input sentences list
	"""

	def make_chunks(l, n):
		"""Yield successive n-sized chunks from l."""
		for i in range(0, len(l), n):
			yield ' '.join(l[i:i + n])

	sentences_split = []
	for sentence in sentences:
		chunks = list(make_chunks(sentence.split(), l))
		sentences_split += chunks

	#print(DELIMITER + 'After sentence splitting:')
	#print(sentences_split[:2])
	return sentences_split


def extract_sibling_sentences(sentences):
	'''
	Parse sentence POS tree and extract a list of noun phrases
	:param sentences: List<String> of sentences to extract from
	:return: List<String> of all sub sentences
	'''
	# todo

	parsed_sents = [list(stan_parser.raw_parse(sent))[0] for sent in sentences]
	sub_sents = []
	for sent in parsed_sents:
		sub_sub = []
		for sub_sent in sent[0]:
			if sub_sent.label() == 'S':
				sub_sub.append(' '.join(sub_sent.leaves()) + '.')
		if len(sub_sub) == 0:
			sub_sents.append(' '.join(sent.leaves()))
		else:
			sub_sents += sub_sub
	return sub_sents

def do_stemming(sentences):
	"""
	Stem sentences using Porter stemmer
	:param sentences: list of sentences to stem
	:return: list of stemmed sentences
	"""
	stemmer = PorterStemmer()

	stemmed_sentences = []
	for sentence in sentences:
		words = word_tokenize(sentence)

		stemmed_sentence = []
		for w in words:
			stemmed_sentence.append(stemmer.stem(w))
		stemmed_sentences.append(' '.join(stemmed_sentence))

	return stemmed_sentences


def do_lemmatization(sentences):
	"""
	Run NLTK lemmatization on sentences
	:param sentences: List of sentences
	:return: List of lemmatized sentences
	"""
	wlem = WordNetLemmatizer()

	lemma_sentences = []
	for sentence in sentences:
		words = word_tokenize(sentence)

		lemma_sentence = []
		for w in words:
			lemma_sentence.append(wlem.lemmatize(w))
		lemma_sentences.append(' '.join(lemma_sentence))

	#print(DELIMITER + 'After lemmatization:')
	#print(lemma_sentences[:2])
	return lemma_sentences


# Todo - only removes first stopword of bigram
def remove_stopword_bigrams(sentences):
	"""
	Removes bigrams that consist of only stopwords, as suggested by Yogotama et al
	:param sentences: list of sentences
	:return: list of sentences with stopword bigrams removed
	"""
	sw_bigrams_removed = []
	for sentence in sentences:
		bigrams = nltk.bigrams(sentence.split())
		stopwords = nltk.corpus.stopwords.words('english')

		filtered = [tup for tup in bigrams if not [True for wrd in tup if wrd in stopwords].count(True) == 2]
		# Join back into sentence:
		joined = " ".join("%s" % tup[0] for tup in filtered)
		sw_bigrams_removed.append(joined)

	#print(DELIMITER + 'After removing stopword bigrams:')
	#print(sw_bigrams_removed[:2])
	#print(len(sw_bigrams_removed))
	return sw_bigrams_removed


def vectorize(sentences, ngram_range=(1,1), tfidf=False):
	"""
	Vectorize sentences using plain sklearn.feature_extraction.CountVectorizer.
	Represents the corpus as a matrix of sentences by word counts.
	:param sentences: list of sentences
	:return: scipy.sparse.coo_matrix - vectorized word counts. N (# sentences) x M (length of vocabulary)
	"""
	if tfidf:
		vectorizer = TfidfVectorizer(ngram_range=ngram_range)
	else:
		vectorizer = CountVectorizer(ngram_range=ngram_range)
	vectorizer.fit(sentences)

	transformed = vectorizer.transform(sentences)
	# print(DELIMITER + 'After vectorization (ngram_range: {}):'.format(ngram_range))
	# print(transformed.shape)
	# print(transformed[:2])

	return transformed


def extract_noun_phrases(sentences):
	'''
	Parse sentence POS tree and extract a list of noun phrases
	:param sentences: List<String> of sentences to extract from
	:return: List<String> of all noun phrases
	'''
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	sentences = [nltk.pos_tag(sent) for sent in sentences]

	grammar = "NP: {<DT>?<JJ>*<NN>}"

	cp = nltk.RegexpParser(grammar)
	result = [cp.parse(sentence) for sentence in sentences]

	noun_phrases = []
	for s in result:
		for r in s:
			if isinstance(r, nltk.tree.Tree):
				if r.label():
					if r.label() == 'NP':
						nps = [word for word, tag in r.leaves()]
						noun_phrases.append(' '.join(nps))

	return noun_phrases

def do_exclude_misspelled(sentences):
	sentences_spellchecked = []
	for sentence in sentences:
		# for w in word_tokenize(sentence):
			# if ENCHANT_DICT.check(w) == False:
            #
			# 	sentence.replace(w, '')

		# Make sure we didn't remove all words from the vector
		pattern = re.compile(r'\S')
		if pattern.match(sentence):
			sentences_spellchecked.append(sentence)

	print(DELIMITER + 'After excluding misspelled:')
	print(sentences_spellchecked[:2])
	return sentences_spellchecked