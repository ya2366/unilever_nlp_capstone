import os
import io
import json
import scipy
import logging
import sklearn
import pandas as pd
import scipy.sparse
import sklearn.metrics.pairwise
from sklearn.preprocessing import Normalizer

#from scipy.sparse import csr_matrix
from preprocessing import *
from scipy.sparse import lil_matrix


from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from preprocessing import *
import numpy as np

DELIMITER = '\n' + '*' * 30 + ' '
csr_matrix.__hash__ = object.__hash__


def ortho_proj_vec(vectors, B):
    """
    Find the vector that is furthest from the subspace spanned by the vectors in B
    :param vectors: List of vectors to search
    :param B: Set of unit vectors that form the basis of the subspace
    :return: Index of furthest vector
    """
    projs = lil_matrix(vectors.shape, dtype=np.int8)  # try coo_matrix?

    for b in B:
        p_i = np.multiply(vectors.dot(b.T), b)
        projs += p_i

    dists = scipy.sparse.linalg.norm((vectors - projs), axis=1)

    return np.argmax(dists)


def compute_mean_vector(vectors):

    c = np.mean(vectors, axis=0, dtype='float64', out=None)

    return lil_matrix(c)


def compute_primary_basis_vector(vectors, sentences, d, L):
    """
    Find the vector in vectors with furthest distances from the given vector d, while the
    length of the corresponding sentence from sentences does not overflow the word limit L
    :param vectors: List of vectorized sentences to search through
    :param sentences: Corresponding list of sentences
    :param d: Test vector to compare to each vector in vectors
    :param L: Max length of word count in sentence
    :return: Index of vector with largest distance in vectors
    """

    L = int(L)
    dists = sklearn.metrics.pairwise.pairwise_distances(vectors, d)  # should be proj_b0^ui ?

    p = np.argmax(dists)

    # Skip vectors that overflow the word limit
    total_length = len(sentences[p].split())
    # Include length of first vector if we aren't dealing with `d` as the mean vector. Todo

    while total_length > L:
        vectors[p] = np.zeros(vectors[p].shape)
        dists = sklearn.metrics.pairwise.pairwise_distances(vectors, d)
        p = np.argmax(dists)
        total_length = len(sentences[p].split())

        if type(d) != lil_matrix:
            if type(d) != scipy.sparse.csr.lil_matrix:
                total_length += len(sentences[d].split())

    return p


def sem_vol_max(sentences, vectors, L):
    """
    Perform Semantic Volume Maximization as specified in Yogotama et al.
    :param sentences: List of original, un-preprocessed sentences
    :param vectors: np.array of vectorized sentences corresponding to sentences
    :param L: Limit of number of words to include in summary
    :return:
    """
    S = set()
    B = set()

    # Mean vector
    L = int(L)
    c = compute_mean_vector(vectors)

    # 1st furthest vector -- Will this always just be the longest sentence?
    p = compute_primary_basis_vector(vectors, sentences, c, L)
    vec_p = vectors[p]
    sent_p = sentences[p]
    S.add(sent_p)

    # 2nd furthest vector
    q = compute_primary_basis_vector(vectors, sentences, vec_p, L)
    vec_q = vectors[q]
    sent_q = sentences[q]
    S.add(sent_q)

    b_0 = vec_q / scipy.sparse.linalg.norm(vec_q)

    print(b_0)
    print(type(b_0))

    print("vec_q", vec_q)
    print("vec_q_type", type(vec_q))
    print("b_0", b_0)

    B.add(b_0)

    return sentence_add_loop(vectors, sentences, S, B, L)


def sentence_add_loop(vectors, sentences, S, B, L):
    """
    Iteratively add most distance vector in `vectors`, from list of vectors `vectors`, to set B.
    Add the corresponding sentences from `sentences` to set S.
    :param vectors: Vectors to search through
    :param sentences: Corresponding sentences to add to set
    :param S: Set of sentences to be returned
    :param B: Set of existing basis vectors of the subspace to be compared to each new vector
    :param L: Max length of total number of words across all sentences in S
    :return: List of sentences corresponding to set of vectors in `vectors`
    that maximally span the subspace of `vectors`
    """
    L = int(L)
    total_length = sum([len(sent.split()) for sent in S])
    exceeded_length_count = 0

    for i in range(0, vectors.shape[0]):
        r = ortho_proj_vec(vectors, B)

        new_sentence_length = len(sentences[r].split())

        # Todo - norm may be zero if sentence only had stopwords
        norm = scipy.sparse.linalg.norm(vectors[r])

        if total_length + new_sentence_length <= L and norm != 0:
            b_r = np.divide(vectors[r], norm)

            S.add(sentences[r])
            B.add(b_r)

            total_length += new_sentence_length
            # Reset the exceeded_length_count

            exceeded_length_count = 0
            # Prevent us from adding this sentence again
            # Todo - original authors had this same problem?
            vectors[r] = np.zeros(vectors[r].shape)

        else:
            vectors[r] = np.zeros(vectors[r].shape)

            exceeded_length_count += 1
            if exceeded_length_count >= 15:
                break

    # print("Final sentence count: " + str(len(S)))
    return [str(e) for e in S]

def summarize(
        data,
        columns=[],
        l=100,
        ngram_range=(2, 3),
        tfidf=True,
        use_svd=True,
        k=10,
        scale_vectors=True,
        use_noun_phrases=False,
        extract_sibling_sents=False,
        split_longer_sentences=True,
        to_split_length=50,
        exclude_misspelled=True):
    """
    Start summarization task on excel file with columns to summarize
    :param data: String - Name of excel file with columns to summarize
        or pandas.DataFrame - DataFrame to summarize by column
    :param columns: List<String> - Titles in first row of spreadsheet for columns to summarize
    :param l: Integer - Max length of summary (in words)
    :param use_bigrams: Boolean - Whether to summarize based on word counts or bigrams
    :param use_svd: Boolean - Whether to summarize based on top k word concepts
    :param k: Integer - Number of top word concepts to incorporate
    :param extract_sibling_sents: Boolean - whether to split sentences into individual siblings as defined by adjacent S tags in nltk parse tree
    :param split_long_sentences: Boolean - whether to split longer sentences into shorter ones to prevent bias in distance calculation
    :param to_split_length: Integer - Length above which to split sentences
    :return: List of lists of summary sentences
    """

    summaries = []
    data = data.split(".")
    # Iterate over sentence groups for each column and summarize each

    for i, sentence_set in enumerate([data]):

        if split_longer_sentences:
            sentence_set = split_long_sentences(sentence_set, to_split_length)
            print (1)
        if exclude_misspelled:
            sentence_set = do_exclude_misspelled(sentence_set)
            print (2)
        if extract_sibling_sents:
            sentence_set = extract_sibling_sentences(sentence_set)
            print (3)

        vectors = do_lemmatization(sentence_set)
        vectors = remove_stopword_bigrams(vectors)

        vectors = vectorize(vectors, ngram_range=ngram_range, tfidf=tfidf)

        if scale_vectors:
            normalizer = Normalizer()
            vectors = normalizer.fit_transform(vectors)

        if use_svd:
            vectors = vectors.asfptype()
            print(vectors.shape)
            print(min(vectors.shape))
            print(k)
            if k >= min(vectors.shape):
                print("k too large for vectors shape, lowering...")
                k = min(vectors.shape) - 1

            U, s, V = scipy.sparse.linalg.svds(vectors, k=k)

            print(DELIMITER + 'After SVD:')
            print("U: {}, s: {}, V: {}".format(U.shape, s.shape, V.shape))
            vectors = csr_matrix(U)

        summary = sem_vol_max(sentence_set, vectors, l)

        print(DELIMITER + 'Result:')
        print(summary)

        toappend = [summary, []]

        # todo - ugly
        if use_noun_phrases:
            toappend[1] = extract_noun_phrases(sentence_set)

        summaries.append(toappend)

    print("Columns summarized: {}".format(len(summaries)))


    return summaries

# def read_data_from_txt():
#     articles = os.listdir("training")
#     for article in articles:
#         if article == "Dove_Body_Wash.txt":
#             print('Reading articles/' + article)
#             article_file = io.open('training/' + article, 'r')
#             text = article_file.read()
#             print text
#             summary = summarize(text)
#     return summary
# read_data_from_txt()
