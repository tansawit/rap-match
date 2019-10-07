# Thomas Horak (thorames)
# classifier.py
import re
import sys
import csv
import json
# import operator
import spacy
import math
import random
# import numpy as np
import sklearn.ensemble
import sklearn.metrics
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
# from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import KFold
from nltk.tokenize import RegexpTokenizer
# from sklearn.metrics import f1_score
# from sklearn.metrics import average_precision_score
from collections import Counter
from nltk.stem import PorterStemmer
bad_keys = ["VERB_Count", "PUNCT_Count", "PRON_Count", "ADJ_Count", "INTJ_Count", "ADV_Count", "NOUN_Count"]
rap_stopwords = ['nigga', 'bitch', 'money', 'never', 'right', 'cause', 'still', 'could', 'think', 'fuckin', 'every',
                 'really', 'night', 'around', 'better', 'would', 'black', 'fucking', 'young', 'world', 'break', 'little',
                 'everything', 'start', 'watch', 'tryna', 'pussy', 'friend', 'people', 'might', 'everybody', 'motherfucker',
                 'light', 'smoke', 'gettin', 'bring', 'street', 'thing', 'whole', 'straight', 'another', 'getting', 'catch',
                 'always', 'white', 'leave', 'first', 'throw', 'change', 'though', 'nothing']

def train(X, true_X):
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    feat_dicts = []
    idf = {}
    length = len(X)

    for lyric in X:
        fd = {}
        grams = []
        pos_counts = []

        lyric = lyric.split(" ")
        pos_counts = lyric[-7:]
        lyric = " ".join(lyric[:-7])

        lyric = re.sub(r'[^a-zA-Z\d\s]', '', lyric)
        tokens = tokenizer.tokenize(lyric)

        tokens = [t for t in tokens if t not in rap_stopwords]

        temp_tokens = []
        [temp_tokens.append(stemmer.stem(t)) for t in tokens]
        tokens = temp_tokens

        for i in range(len(tokens)):
            if tokens[i] in fd:
                fd[tokens[i]] += 1
                grams.append(tokens[i])
            else:
                fd[tokens[i]] = 1
                grams.append(tokens[i])
            if i < (len(tokens) - 2):
                if (tokens[i] + "_" + tokens[i + 1]) in fd:
                    fd[tokens[i] + "_" + tokens[i + 1]] += 1
                    grams.append(tokens[i] + "_" + tokens[i + 1])
                else:
                    fd[tokens[i] + "_" + tokens[i + 1]] = 1
                    grams.append(tokens[i] + "_" + tokens[i + 1])

        fd["VERB_Count"] = float(pos_counts[0])
        fd["PUNCT_Count"] = float(pos_counts[1])
        fd["PRON_Count"] = float(pos_counts[2])
        fd["ADJ_Count"] = float(pos_counts[3])
        fd["INTJ_Count"] = float(pos_counts[4])
        fd["ADV_Count"] = float(pos_counts[5])
        fd["NOUN_Count"] = float(pos_counts[6])

        true_X.append(' '.join(tokens))

        gram_length = len(grams)
        for key in fd.keys():
            if key not in bad_keys:
                fd[key] = fd[key] / gram_length
        grams = set(grams)
        for gram in grams:
            if gram in idf:
                idf[gram] += 1
            else:
                idf[gram] = 1
        feat_dicts.append(fd)

    for feat_dict in feat_dicts:
        for key in feat_dict.keys():
            if key not in bad_keys:
                IDF = math.log(1 + (length / idf[key]))
                feat_dict[key] = feat_dict[key] * IDF

    return feat_dicts

def processLyrics(lyrics):
    authors = []
    for author in lyrics:
        for song in lyrics[author]:
            lyric = re.sub(r'\[[^>]+\]', '', song["lyrics"])
            lyric = re.sub(r'\([^>]+\)', '', lyric)
            lyric = re.sub(r'\{[^>]+\}', '', lyric)
            lyric = lyric.split(r'\s')
            text = ""
            for line in lyric:
                line = re.sub(r'\n', ' ', line)
                text += (" " + line)
            for PartOfSpeech in song["pos_counts"]:
                if PartOfSpeech == "VERB":
                    verb = song["pos_counts"]["VERB"]
                if PartOfSpeech == "PUNCT":
                    punct = song["pos_counts"]["PUNCT"]
                if PartOfSpeech == "PRON":
                    pron = song["pos_counts"]["PRON"]
                if PartOfSpeech == "ADJ":
                    adj = song["pos_counts"]["ADJ"]
                if PartOfSpeech == "INTJ":
                    intj = song["pos_counts"]["INTJ"]
                if PartOfSpeech == "ADV":
                    adv = song["pos_counts"]["ADV"]
                if PartOfSpeech == "NOUN":
                    noun = song["pos_counts"]["NOUN"]
            text += (" " + str(verb) + " " + str(punct) + " " + str(pron) + " " + str(adj) + " " + str(intj) + " " + str(adv) + " " + str(noun))
            authors.append([author, text])
    return authors

def dectree_main(var):
    y = []
    X = []
    true_X_train = []
    true_X_test = []
    tokenizer = RegexpTokenizer(r'\w+')
    authors = {}
    POS = {}

    with open('preprocessedf_corpus.json') as json_file:
        lyrics = json.load(json_file)

        authors = processLyrics(lyrics)

    for pair in authors:
        y.append(pair[0])
        X.append(pair[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    feature_dicts = train(X_train, true_X_train)

    transform = DictVectorizer()
    X = transform.fit_transform(feature_dicts)
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=16)
    clf.fit(X, y_train)

    var = var.lower()

    nlp = spacy.load('en_core_web_sm')
    songlyrics = nlp(var)
    pos_counts = dict(Counter([token.pos_ for token in songlyrics]))
    if "VERB" not in pos_counts:
        pos_counts["VERB"] = 0
    if "PUNCT" not in pos_counts:
        pos_counts["PUNCT"] = 0
    if "PRON" not in pos_counts:
        pos_counts["PRON"] = 0
    if "ADJ" not in pos_counts:
        pos_counts["ADJ"] = 0
    if "INTJ" not in pos_counts:
        pos_counts["INTJ"] = 0
    if "ADV" not in pos_counts:
        pos_counts["ADV"] = 0
    if "NOUN" not in pos_counts:
        pos_counts["NOUN"] = 0
    var += (" " + str(pos_counts["VERB"]) + " " + str(pos_counts["PUNCT"]) + " " + str(pos_counts["PRON"]) + " " + str(pos_counts["ADJ"]) + " " + str(pos_counts["INTJ"]) + " " + str(pos_counts["ADV"]) + " " + str(pos_counts["NOUN"]))

    X_test.append(var)

    X_test_fd = train(X_test, true_X_test)

    X_test_vecs = transform.transform(X_test_fd)
    y_pred = clf.predict_proba(X_test_vecs)
    array = clf.classes_.tolist()

    best_value = float("-inf")
    best_index = -1
    for i in range(len(y_pred[0])):
        if y_pred[0][i] == best_value:
            random_choice = random.choice([best_index, i])
            best_index = random_choice
        if y_pred[0][i] > best_value:
            best_value = y_pred[0][i]
            best_index = i

    return array[best_index]
