#!/usr/bin/env python
# -*- coding:utf-8 -*-

import nltk
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import string
import sys
import re
 
# nltk.download('punkt')
# nltk.download('stopwords')

def log(message:str, name:str='process'):
    '''Writes a specified message to a log file including timestamp.'''
    print(message)
    # Creating or appending the log file and appending the message:
    file = open(name + '_log.txt', 'a')
    file.write(str(pd.Timestamp.today()) + " " + message + "\n")
    file.close()

def connect(base:str="test.db"):
    '''Connects to an SQL database and either return the connection or an error
    message if no connection could be established.'''
    try:
        # Establishing the SQLite connection to the specified database file:
        connection = sqlite3.connect(base)
        log("Connection established to \"" + base + "\"")
        return connection
    except sqlite3.Error as error:
        # If no connection could be established, print the Error message and save the date in a csv file:
        log("Connection to " + base + " failed: " + str(error) + ", terminating program")
        raise Exception(error)

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
            stop_words = nltk.corpus.stopwords.words('english')
            stop_words.extend(nltk.corpus.stopwords.words('german'))
            if stop:
                stop_words.extend(stop)
            stop_filter = lambda words: [w for w in words if not w in set(stop_words)]
            df.Text = df.Text.apply(stop_filter)
        except:
            log("Warning! Filtering stop-words failed")
        # Stemming words using Snowball stemmer:
        # try:
        #     log("... stemming words")
        #     stem = lambda words: [nltk.stem.snowball.SnowballStemmer('german').stem(word) for word in words]
        #     df.Text = df.Text.apply(stem)
        # except:
        #     log("Warning! Stemming words failed")
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

def freq(documents:list, min_df:float=0.05, max_df:float=0.90) -> list:
    '''Calculates term and document frequencies (TF, relTF, DF, IDF, TFIDF)
    from a list of documents and returns them as lists of dictionaries for each
    of them. Also assigns a rank to each word in a document.'''
    log("Calculating term and document frequencies")
    try:
        # Converting documents to a matrix of TF-IDF features, only considering
        # terms that appear in the mininum and maximum number of documents:
        log("... vectorizing documents and extracting features")
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
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
    except:
        log("Error! Applying count vectorizer failed, program ended after " + str(pd.Timestamp.now() - start))
        raise Exception("sklearn.TfidfVectorizer failed")

    log("... calculating term frequency (TF) and relative term frequency (RTF)")
    # Calculating the term frequency (TF) of each document, i.e. the number of
    # times each term is used in each document:
    try:
        texts = [] # List of all documents as lists of terms
        for text in documents:
            text = text.split()
            texts.append(text)
        
        term_freq = []
        rel_term_freq = []
        rank = []
        for text in texts:
            tf = dict.fromkeys(features, 0) # dictionary containing word frequencies
            for term in features:
                tf[term] = text.count(term)
            term_freq.append(tf)

            # Calculating the relative term frequency (RTF) in percent:
            rtf = {} # dictionary containing relative term frequencies
            for term, frequency in tf.items():
                if len(text) > 0:
                    rtf[term] = frequency / len(text)
                else:
                    rtf[term] = 0
            rel_term_freq.append(rtf)

            # Generating the rank of the term in the text (based on the RTF):
            tmp = {}
            n = 1
            for i in sorted(list(set(rtf.values())), reverse=True):
                tmp[i] = n
                n += 1
            rnk = []
            for i in rtf.values():
                rnk.append(tmp[i])
            rnk = dict(zip(rtf.keys(), rnk))
            rank.append(rnk)

        # Calculating the document frequency (DF) of each term, i.e. the number of
        # documents each term is used in:
        log("... calculating document frequency (DF)")
        df = dict.fromkeys(features, 0)
        for term in features:
            for text in texts:
                if term in text:
                    df[term] += 1
        doc_freq = []
        for _ in range(0, len(documents)):
            doc_freq.append(df)
    except:
        log("Error! Calculating term frequencies failed, program ended after " + str(pd.Timestamp.now() - start))
        raise Exception("Calculating term frequencies failed")

    log("Finished calculating term and document frequencies")
    return term_freq, rel_term_freq, rank, doc_freq, inv_doc_freq, term_freq_inv_doc_freq

if __name__ == "__main__":
    log("Starting text processing")
    start = pd.Timestamp.now()
    # Establishing connection to database:
    conn = connect('stage.db')
    # Reading raw website data from the database table:
    if len(sys.argv) > 1:
        table = str(sys.argv[1])
    else:
        log("Error! No data table specified, program ended after " + str(pd.Timestamp.now() - start))
        raise Exception("Data table name is required as input")
    raw = pd.read_sql("SELECT * FROM raw_" + table, conn)
    log("Reading table \"" + table + "\"")

    # Reading file containing stop words:
    try:
        log("Reading stop words from file \"stop.txt\"")
        file = open('stop.txt', 'r')
        stop = [line.strip("\n") for line in file.readlines()]
        file.close()
    except:
        log("Warning! File containing stop words could not be read")
        stop = []
    
    # Joining and tokenizing all website texts:
    jnd = join(raw)
    token = tokenize(jnd, stop=stop)
    # Storing the tokenized data in the database:
    try:
        token.to_sql("token_" + table, conn, index=False, if_exists='append')
        log("Storing token data as table \"token_" + table + "\"" + " in the database")
    except:
        log("Warning! Storing data in database failed, saving as \"backup_token_" + table + "\" instead")
        token.to_csv("backup_token_" + table + ".csv", index=False, sep='\t')

    # Calculating text features for each text:
    tf, rtf, r, df, idf, tfidf = freq(token.Text)

    # Creating data table of text features:
    try:
        log("Creating data table with features")
        results = pd.DataFrame()
        for i in range(0, len(jnd.Site)):
            frame = pd.DataFrame({
                'Site' : jnd.Site.iloc[i],
                'Date' : jnd.Date.iloc[i],
                'Term' : list(tf[i].keys()),
                'TF' : list(tf[i].values()),
                'rTF' : list(rtf[i].values()),
                'DF' : list(df[i].values()),
                'IDF' : list(idf[i].values()),
                'TFIDF' : list(tfidf[i].values()),
                'Rank' : list(r[i].values())
            }).set_index(['Site', 'Date', 'Term'])
            results = pd.concat([results, frame])
        results = results.reset_index()
    except:
        log("Error! Creading data table failed, program ended after " + str(pd.Timestamp.now() - start))

    # Storing the data and its features in the database:
    try:
        results.to_sql("feat_" + table, conn, index=False, if_exists='append')
        log("Storing token data as table \"feat_" + table + "\"" + " in the database")
    except:
        log("Warning! Storing data in database failed, saving as \"backup_token_" + table + "\" instead")
        results.to_csv("backup_feat_" + table + ".csv", index=False, sep='\t')

    end = pd.Timestamp.now()
    log("Program sucessfully finished after " + str(end - start))
