#!/usr/bin/env python
# -*- coding:utf-8 -*-

import nltk
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import string
import re

from module import connect, log
 
# nltk.download('punkt')
# nltk.download('stopwords')

def join(raw:pd.DataFrame) -> pd.DataFrame:
    '''Joins the text content of multiple subsites of the same main website
    together in one string, removing newlines and redundant empty spaces.'''
    log("Joining texts for websites")
    # Creating a new dataframe for main sites only:
    try:
        join = []
        for url in raw.Main.unique():
            df = raw.loc[raw.Main == url]
            main = df.Main.unique()[0]
            date = df.Date.min()
            text = []
            # Add titles to the site text string:
            text.extend([ttl for ttl in df.Title])
            # Add all other text to the string, replacing newlines and spaces
            for txt in df.Text:
                text.extend([paragraph for paragraph in txt.split("\r\n")])
            content = " ".join([txt.replace("\n", " ").replace("\xa0", " ") for txt in text])
            join.append([main, content, date])
        return pd.DataFrame(join, columns=('Site', 'Text', 'Date'))
    except:
        log("Error! Joining texts failed, program ended after " + str(pd.Timestamp.now() - start))
        return Exception("Failed joining texts")

def tokenize(df:pd.DataFrame, stop:list=None) -> pd.DataFrame:
    '''Creates word tokens from a list or series of texts in a data table,
    removing words with thre or less letters, punctuation, non-alphabetic
    and non-latin words, converting words to lower-case, filtering stop-words
    and stemming the words.'''
    log('Starting tokenization and word transformation')
    # Create list of words as tokens:
    try:
        log("Creating word tokens")
        df.Text = df.Text.apply(nltk.tokenize.word_tokenize)
    except:
        log("Error! Creating word tokens failed, program ended after " + str(pd.Timestamp.now() - start))
        return Exception("Failed creating tokens")
    else:
        # Remove words with 3 or less letters:
        try:
            log("... removing words with three or less letters")
            three_less = lambda words: [word for word in words if len(word) > 3]
            df.Text = df.Text.apply(three_less)
        except:
            log("Warning! Removing words with three or less letters failed")
        # Remove all punctuation:
        try:
            log("... removing punctuation from words")
            table = str.maketrans('', '', string.punctuation)
            punctuation = lambda words: [w.translate(table) for w in words]
            df.Text = df.Text.apply(punctuation)
        except:
            log("Warning! Removing punctuation failed")
        # Remove all words that contain non-alphabetic letters:
        try:
            log("... removing words containing non-alphabetic letters")
            non_alpha = lambda words: [word for word in words if word.isalpha()]
            df.Text = df.Text.apply(non_alpha)
        except:
            log("Warning! Removing non-alphabetic tokens failed")
        # Remove all non-latin words:
        try:
            log("... removing non-latin words")
            nl = r"[^\\x00-\\x7F\\x80-\\xFF\\u0100-\\u017F\\u0180-\\u024F\\u1E00-\\u1EFF]"
            non_latin = lambda words: [word for word in words if re.match(nl, word) == None]
            df.Text = df.Text.apply(non_latin)
        except:
            log("Warning! Removing non-latin tokens failed")
        # Make words lower case:
        try:
            log("... transforming words to lower-case")
            lower_case = lambda words: [word.lower() for word in words]
            df.Text = df.Text.apply(lower_case)
        except:
            log("Warning! Transforming to lower-case failed")
        # Filter out stop words from the NLTK corpus and an optional list:
        try:
            log("... filtering stop-words")
            stop_words = nltk.corpus.stopwords.words('german')
            if stop:
                stop_words.extend(stop)
            stop_filter = lambda words: [w for w in words if not w in set(stop_words)]
            df.Text = df.Text.apply(stop_filter)
        except:
            log("Warning! Filtering stop-words failed")
        # Stemming words using Snowball stemmer:
        try:
            log("... stemming words")
            stem = lambda words: [nltk.stem.snowball.SnowballStemmer('german').stem(word) for word in words]
            df.Text = df.Text.apply(stem)
        except:
            log("Warning! Stemming words failed")
        # Removing non-ascii words that could not be stemmed (i.e. foreign language words)
        try:
            log("... removing non-ASCII words")
            non_ascii = lambda words: [word for word in words if not any(ord(letter) > 127 for letter in word)]
            df.Text = df.Text.apply(non_ascii)
        except:
            log("Warning! Removing non-ASCII words failed")
        # Remove stemmed words with less than 3 letters:
        try:
            log("... removing words with three or less letters")
            three_less = lambda words: [word for word in words if len(word) >= 3]
            df.Text = df.Text.apply(three_less)
        except:
            log("Warning! Removing words with three or less letters failed")
        log("Tokenization and word transformation complete")
        join = lambda text: " ".join(text)
        df.Text = df.Text.apply(join)
    return df

def freq(documents:list) -> dict:
    '''Calculates term and document frequencies (TF, relTF, DF, IDF, TFIDF)
    from a list of documents and returns them as lists of dictionaries for each
    of them.'''
    log("Calculating term and document frequencies")
    terms = [] # List of all terms
    doc_terms = [] # List of all documents as lists of terms
    log("... splitting documents and creating set of terms")
    for doc in documents:
        doc = doc.split()
        doc_terms.append(doc)
        terms.extend(doc)
    # Creating a sorted list of all unique terms in all documents:
    terms = list(set(terms))
    terms.sort()

    # Calculating the term frequency (TF) of each document, i.e. the number of
    # times each term is used in each document:
    log("... calculating term frequency (TF) and relative term frequency (RTF)")
    term_freq = []
    rel_term_freq = []
    for doc in doc_terms:
        tf = dict.fromkeys(terms, 0) # dictionary containing word frequencies
        for term in doc:
            tf[term] += 1
        term_freq.append(tf)

        # Calculating the relative term frequency (RTF) in percent:
        rtf = {} # dictionary containing relative term frequencies
        size = len(doc)
        for t, f in tf.items():
            rtf[t] = f/size
        rel_term_freq.append(rtf)

    # Calculating the document frequency (DF) of each term, i.e. the number of
    # documents each term is used in:
    log("... calculating document frequency (DF)")
    df = dict.fromkeys(term_freq[0], 0)
    for term in df:
        for d in term_freq:
            if d[term] > 0:
                df[term] += 1
    doc_freq = []
    for _ in range(0, len(term_freq)):
        doc_freq.append(df)

    log("... vectorizing documents and extracting features")
    # Converting documents to a matrix of TF-IDF features:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)
    # Creating list of features (= terms):
    features = vectorizer.get_feature_names()

    # Calculating inverse document frequency (IDF) using the vectorizer:
    inv_doc_freq = []
    log("... calculating inverse document frequency (IDF)")
    for _ in range(0,len(documents)):
        inv_doc_freq.append(dict(zip(features, vectorizer.idf_)))

    # Calculating term frequency inverse document frequency (TFIDF):
    term_freq_inv_doc_freq = []
    log("... calculating term frequency inverse document frequency (TFIDF)")
    for tfidf in vectorizer.transform(documents).toarray():
        term_freq_inv_doc_freq.append(dict(zip(features, tfidf)))

    log("Finished calculating term and document frequency")
    return term_freq, rel_term_freq, doc_freq, inv_doc_freq, term_freq_inv_doc_freq

if __name__ == "__main__":
    log("Starting text processing")
    start = pd.Timestamp.now()
    # Reading raw website data from the database table:
    table = 'Startups100'
    raw = pd.read_sql("SELECT * FROM raw_" + table, connect(base='Stage.db'))
    # Joining and tokenizing all website texts:
    jnd = join(raw)
    token = tokenize(jnd)
    # Storing the tokenized data in the database:
    try:
        token.to_sql("token_" + table, connect(base='Stage.db'), index=False, if_exists='append')
        log("Storing token data as table \"token_" + table + "\"" + " in the database")
    except:
        log("Warning! Storing data in database failed, saving as \"backup_token_" + table + "\" instead")
        token.to_csv("backup_token_" + table + ".csv", index=False, sep='\t')

    # Calculating text features for each text:
    tf, rtf, df, idf, tfidf = freq(token.Text)

    # Creating data tables of text features:
    log("Creating data table with text features")
    results = pd.DataFrame()
    for i in range(0, len(jnd.Site)):
        frame = pd.DataFrame({
            'Site' : jnd.Site.iloc[i],
            'Date' : jnd.Date.iloc[i],
            'Term' : list(tf[i].keys()),
            'TF' : list(tf[i].values()),
            'rTF' : list(rtf[i].values()),
            'DF' : list(tf[i].values()),
            'IDF' : list(idf[i].values()),
            'TFIDF' : list(tfidf[i].values()),
        }).set_index(['Site', 'Date', 'Term'])
        results = pd.concat([results, frame])
    results = results.reset_index()

    # Storing the features of the data in the database:
    try:
        results.to_sql("feat_" + table, connect(base='Stage.db'), index=False, if_exists='append')
        log("Storing token data as table \"feat_" + table + "\"" + " in the database")
    except:
        log("Warning! Storing data in database failed, saving as \"backup_token_" + table + "\" instead")
        results.to_csv("backup_feat_" + table + ".csv", index=False, sep='\t')

    end = pd.Timestamp.now()
    log("Program ended after " + str(end - start))
