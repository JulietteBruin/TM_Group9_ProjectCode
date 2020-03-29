import utils
import gensim
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random

def training_instances(descriptions):
    instances = []
    topic_modelling = utils.topic_modelling(descriptions)
    sentiment_score = utils.vader_description(descriptions)

    def Merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res

    count = 0
    for description in sentiment_score:
        dict = Merge(description, topic_modelling[count])
        count += 1
        instances.append(dict)
    return instances

def topic_modelling(descriptions):
    topics = []
    dictionary = gensim.corpora.Dictionary(descriptions)

    bow_corpus = [dictionary.doc2bow(description) for description in descriptions]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=100, id2word=dictionary, passes=2, workers=2)
    for sentence in bow_corpus:
        dict = {}
        value_topic = "".join(repr(e) for e in lda_model.get_document_topics(sentence))
        if value_topic[1:value_topic.find(',')] == '':
            topic_nr = random.randrange(0,100)/100;
        else:
            topic_nr = int(value_topic[1:value_topic.find(',')])/100

        dict['topic'] = topic_nr
        topics.append(dict)

    return topics


vader_model = SentimentIntensityAnalyzer()
def vader_description(descriptions):
    sentiments = []
    for sentence in descriptions:
        sent = ''
        for token in sentence:
            sent += ' ' + token
        vader = vader_model.polarity_scores(sent)
        sentiments.append(vader)
    return sentiments

#splits decriptions into tokens; returns list of lists of tokens
def split_description(descriptions):
    tokenized_descriptions = []
    for description in descriptions:
        sentences_nltk = sent_tokenize(description)
        tokens_per_description = []
        for sentence_nltk in sentences_nltk:
            tokens_per_description = word_tokenize(sentence_nltk)
            tokens_per_description = remove_punctuation(tokens_per_description)
            tokens_per_description = apply_lemma(tokens_per_description)
            #maybe don't apply because it could be meaningful
            #tokens_per_description = remove_stopwords(tokens_per_description)
        tokenized_descriptions.append(tokens_per_description)
    return tokenized_descriptions


def remove_stopwords(description):
    english_stopwords = stopwords.words('english')
    set_english_stopwords = set(english_stopwords)
    without_stopwords = []
    for token in description:
        if token.lower() not in set_english_stopwords:
            without_stopwords.append(token)
    return without_stopwords


def remove_punctuation(description):
    without_punctuation = []
    for token in description:
        translation = token.translate({ord(char): '' for char in string.punctuation})
        if translation != '':
            without_punctuation.append(translation)
    return without_punctuation

def apply_lemma(description):
    lemmatized_description = []
    wordnet = WordNetLemmatizer()
    for token in description:
        lemmatized_description.append(wordnet.lemmatize(token))
    return lemmatized_description
