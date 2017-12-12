import json
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import text_rank_summary
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

def spell_check(sentence_list):
    ## the input is a list of sentences
    with open('dico_spell.json', 'r') as f:
        dico_spell = json.load(f)

    for i in range(len(sentence_list)):
        sentence_split = sentence_list[i].split(' ')
        for k in range(len(sentence_split)):
            if sentence_split[k] in dico_spell.keys():
                sentence_split[k] = dico_spell[sentence_split[k]]
        sentence_list[i] = str(' '.join(sentence_split))

    return sentence_list

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

def do_stem_for_single_sentence(sentence):
    stemmer = PorterStemmer()
    words = word_tokenize(sentence)
    stemmed_sentence = []
    for w in words:
        stemmed_sentence.append(stemmer.stem(w))
    return " ".join(stemmed_sentence)


def handle_neg(candidate):
    candidate = candidate.lower()
    neg_items = ["don't", "doesn't",'not a lot of', 'not', 'no', 'non', 'not', 'nor', 'free of', 'not too', 'not to', 'clear of']
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

good_attitude = ["nice", "great", "good", "like", "love", "well", "fan", "recommend"]
bad_attitude = ["try something else"] # or negation word before good_attitudes
adverb = ["big", "small", "little", "deep", "long", "use"]
data = "Moisturizing body wash for winter dry skin. If you are sensitive to scented bath products,you might want to try something else. Consistently leaves my skin, and feeling soft and clean. The smell is nice. I've been using Dove's sensitive skin body wash for years. of the bottle. I thought that was the feeling of clean. Great product. Overall, it's good body wash and I really like it! It lathers well and a little goes a long way. This body wash is great. Dove Sensitive Skin Nourishing Body Wash. Nourishing Moisture. I love this body wash. I do feel nicely moisturized after using the product. This is your standard body wash. It lathers well and has a light, clean scent. I like the scent of the deep moisture a lot better than the scent of the sensitive skin. It is not. I have used Dove products for years. I really like this soap. It smells great and doesn't irritate. I already use Dove products. Love it. I am a big fan of Dove beauty bars. Dove Body Wash with nourishing Moisture, Deep Moisture Nourishing. It is very moisturizing. Highly recommend. I love this product. (Especially in the winter months!) "


#data = data.split(".")
# data = spell_check(data)
# res = pos_mark(tokenization("".join(data)))

def get_attributes(data_list):
    candidates = set()
    #data_list = data_list[0]

    for data in data_list:
        for item in data:
            if (item[1] == "JJ") and (item[0] not in good_attitude and item[0] not in bad_attitude) and len(item[0]) > 1 and item[0] not in adverb:
                candidates.add(item[0])
    return candidates



def summary_attitude(data, good, bad):
    attitude = ""
    for i in range(len(data)):
        for word in data[i].split():
            if word in good_attitude:
                good += 1
            elif word in bad_attitude or word[:3] == "no_":
                bad += 1
    if bad == 0 and good == 0:
        attitude = "The product is fairly good."
    elif bad == 0:
        attitude = "The product is highly recommended."
    elif good / bad > 5:
        attitude = "The product is highly recommended."
    elif good / bad > 3:
        attitude = "The product can be recommended to customers"
    elif good / bad >= 1:
        attitude = "The product is fairly good."
    elif good < bad:
        attitude = "The product is not recommended."
    return attitude

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    score /= count
    return score

# sentences = [
#     "Dogs are awesome.",
#     "Some gorgeous creatures are felines.",
#     "Dolphins are swimming mammals.",
#     "Cats are beautiful animals.",
# ]
#
# focus_sentence = "Cats are beautiful animals."
#
# for sentence in sentences:
#     print "Similarity(\"%s\", \"%s\") = %s" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence))
#     print "Similarity(\"%s\", \"%s\") = %s" % (sentence, focus_sentence, sentence_similarity(sentence, focus_sentence))
#     print


def generate(data):

    t = text_rank_summary.extract_sentences(data)
    t_list = t.split()
    for i in range(len(t_list)):
        if  t_list[i] == "I" or t_list[i] == "i":
            t_list[i] = "Customers"

    data = data.split(".")
    data = spell_check(data)
    temp = pos_mark(tokenization(" ".join(data)))
    #print temp
    candidates = get_attributes(temp)
    candidates = list(set(candidates))
    contain = set()
    stemmer = PorterStemmer()
    for j in candidates:
        j = j.lower()
        contain.add(str(stemmer.stem(j)))

    candidates = list(contain)
    ## print candidates
    temp_list = set()
    for single_sentence in data:
        single_sentence = single_sentence.lower()
        flag = True
        for word in good_attitude + bad_attitude:
            if word in single_sentence:
                flag = False
        if flag:
            temperate = do_stem_for_single_sentence(single_sentence)

            for single in temperate.split(" "):
                for w in candidates:
                    if single == w:
                        #print temperate
                        temp_list.add(single_sentence)
                        candidates.remove(single)
            #print candidates
            # for can in candidates:
            #     temperate = do_stem_for_single_sentence(single_sentence)
            #     if can in temperate:
            #         temp_list.add(single_sentence)
            #         candidates.remove(can)
            #         print candidates

    temp_list = list(temp_list)
    # for word in candidates:
    #     if word not in good_attitude and word not in bad_attitude:
    #         for single_sentence in data:
    #             if word in single_sentence:
    #
    #                 for i in single_sentence.split(" "):
    #                     print i
    #                     if i not in good_attitude + bad_attitude:
    #                         temp_list.add(single_sentence + ".")
    #                         break

    data = do_stemming(data)
    for i in range(len(data)):
        data[i] = handle_neg(data[i])
    attitude = summary_attitude(data, 0, 0)
    res = ""
    for i in temp_list:
        res = res + i  + "."
    #print attitude + " " + res
    return attitude + " " + res

#print generate(data)
