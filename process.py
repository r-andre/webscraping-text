#!/usr/bin/env python
# -*- coding:utf-8 -*-

import nltk
import sqlite3
import pandas as pd
import numpy as np
import string
import re
 
# nltk.download('punkt')
# nltk.download('stopwords')

def log(message:str):
    '''Writes a specified message to a log file including timestamp.'''
    print(message)
    # Creating or appending the log file and appending the message:
    file = open('process_log.txt', 'a')
    file.write(str(pd.Timestamp.today()) + " " + message + "\n")
    file.close()

def connect(base:str="stage.db"):
    try:
        # Establishing the SQLite connection to the specified database file:
        connection = sqlite3.connect(base)
        log("Connection established to \"" + base + "\"")
        return connection
    except sqlite3.Error as error:
        # If no connection could be established, print the Error message and save the date in a csv file:
        log("Connection to " + base + " failed: " + str(error) + ", program ended after " + str(pd.Timestamp.now() - start))
        raise Exception(error)

def join(raw:pd.DataFrame) -> pd.DataFrame:
    log("Joining texts for websites")
    summarized = []

    try:
        for url in raw.Main.unique():
            df = raw.loc[raw.Main == url]
            main = df.Main.unique()[0]
            date = df.Date.min()
            text = []
            text.extend([ttl for ttl in df.Title])
            for txt in df.Text:
                text.extend([paragraph for paragraph in txt.split("\r\n")])
            content = " ".join([txt.replace("\n", " ").replace("\xa0", " ") for txt in text])
            summarized.append([main, content, date])
        return pd.DataFrame(summarized, columns=('Site', 'Text', 'Date'))
    except:
        log("Error! Joining texts failed, program ended after " + str(pd.Timestamp.now() - start))
        return Exception("Failed joining texts")

def tokenize(joined:pd.DataFrame, stop:list=None) -> pd.DataFrame:
    log('Starting tokenization and word transformation')
    df = joined.copy()
    # Create list of words as tokens:
    try:
        log("Creating word-tokens")
        df.Text = df.Text.apply(nltk.tokenize.word_tokenize)
    except:
        log("Error! Creating word-tokens failed, program ended after " + str(pd.Timestamp.now() - start))
        return Exception("Failed creating tokens")
    else:
        # Remove words with 3 or less letters:
        try:
            log("Removing words with three or less letters")
            three_less = lambda words: [word for word in words if len(word) > 3]
            df.Text = df.Text.apply(three_less)
        except:
            log("Warning! Removing words with three or less letters failed")
        # Remove all punctuation:
        try:
            log("Removing punctuation from words")
            table = str.maketrans('', '', string.punctuation)
            punctuation = lambda words: [w.translate(table) for w in words]
            df.Text = df.Text.apply(punctuation)
        except:
            log("Warning! Removing punctuation failed")
        # Remove all words that contain non-alphabetic letters:
        try:
            log("Removing words containing non-alphabetic letters")
            non_alpha = lambda words: [word for word in words if word.isalpha()]
            df.Text = df.Text.apply(non_alpha)
        except:
            log("Warning! Removing non-alphabetic tokens failed")
        # Make words lower case:
        try:
            log("Transforming words to lower-case")
            lower_case = lambda words: [word.lower() for word in words]
            df.Text = df.Text.apply(lower_case)
        except:
            log("Warning! Transforming to lower-case failed")
        # Filter out stop words from the NLTK corpus and an optional list:
        try:
            log("Filtering stop-words")
            stop_words = lambda words: [w for w in words if not w in set(nltk.corpus.stopwords.words('german'))]
            if stop:
                stop_words.extend(stop)
            df.Text = df.Text.apply(stop_words)
        except:
            log("Warning! Filtering stop-words failed")
        # Stemming words using Snowball stemmer:
        try:
            log("Stemming words")
            stem = lambda words: [nltk.stem.snowball.SnowballStemmer('german').stem(word) for word in words]
            df.Text = df.Text.apply(stem)
            log("Tokenization complete")
        except:
            log("Warning! Stemming words failed")
    return df

def feature():
    pass

log("Starting text processing")
start = pd.Timestamp.now()
connection = connect()

table = 'Startups100'
raw = pd.read_sql("SELECT * FROM raw_" + table, connection)

joined = join(raw)
tokenized = tokenize(joined, stop=['gmbh', 'impressum'])

log("Storing tokenized data as \"tokenized_)" + table + "\"")
tokenized.to_sql("token_" + table, connection, index=False, if_exists='append')
