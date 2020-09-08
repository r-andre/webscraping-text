#!/usr/bin/env python

'''
This script scrapes a list of websites in a txt file for their text content
using BeautifulSoup 4 and stores the results in a csv file or SQLite database.
It also scrapes the first level of sub-sites (list of href on the main site)
for text. A log of all processes is collected in a separate txt file.
'''

import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import numpy as np
import time
import sys
import re

def log(message:str):
    '''Writes a specified message to a log file including timestamp.'''
    print(message)
    # Creating or appending the log file and appending the message:
    file = open('log.txt', 'a')
    file.write(str(pd.Timestamp.today()) + " " + message + "\n")
    file.close()

def scrape(url:str, main:bool):
    '''Takes all text content of a website for an input URL and stores it in a
    global variable. May also store all links to sub-sites on the main page in
    a separate list in order to scrape them later.'''
    try:
        # Establishing the connection:
        response = requests.get(url, timeout=3)
    except:
        # Returning an error message if no connection could be established:
        log("Error! Response timeout for " + url) # log message
    else:
        # Checking for available access to the site:
        if int(response.status_code / 100) == 2: # response codes 2xx = positive response
            # If access is available, scrape the site:
            soup = BeautifulSoup(response.text, "lxml")
            try:
                print("Scraping " + url)
                # Function to clean the whitespace of the site:
                clean = lambda txt: txt.replace("\t", "").replace("\n", "").replace("\r", "").replace("\xa0", " ")
                # Storing the title of the site:
                title = clean(soup.title.string)
                # Storing the text of the site:
                text = clean(soup.get_text(" ", strip=True))
            except:
                # If no text could be scraped, return an error message:
                log("Error! No text could be scraped for " + url) # log message
            else:
                # Append the scraped information to the list of contents:
                timestamp = str(pd.Timestamp.today().date())
                contents.append((re.search("[^/]*//[^/]*/", url).group(), url, title, text.replace(title, ""), timestamp))

            if main:
                # If the scraped site is the main site the also collect all available links to sub-sites:
                for a in soup.findAll('a', href=True):
                    href = a['href']
                    if href.startswith("/") and len(href) > 1:
                        hrefs.append(re.search("[^/]*//[^/]*", url).group() + href)
                        # Note: re.search is used to sidestep problems of some multi-language sites e.g. that
                        # are accessed with a /de/ version of their site, causing href links to then refer to
                        # /de/de/... and 404 errors; i.e. https://www.website.com/de/ -> https://www.website.com

        else:
            # If access to the site was denied, return an error message:
            log("Error! Response code " + str(response.status_code) + " for " + url) # log message

def db(df:pd.DataFrame, table:str='Test'):
    '''Connects to or creates and SQLite database and stores all scraped
    content in the specified table. If this process fails, the data is backed
    up in an csv file.'''
    base = "contents_sqlite.db"
    try:
        # Establishing the SQLite connection to the specified database file:
        connection = sqlite3.connect(base)
        # Writing the data to the specified table:
        df.to_sql(table, connection, index=False, if_exists='append')
        log("Connection established to " + base + ", storing scraped data in table " + table)
    except sqlite3.Error as error:
        # If no connection could be established, print the Error message and save the date in a csv file:
        log("Connection to " + base + " failed: " + str(error))
        filename = 'backup.csv'
        log("Saving scraped data as " + filename)
        df.to_csv(filename, index=False, sep="\t")

def validate(df:pd.DataFrame):
    '''Checks if the scraped data is valid and as expected.'''
    if df.empty:
        # End the program if no data was scraped
        log("No data was scraped, terminating program")
        raise Exception("Validity check failure: empty dataframe")
    else:
        log("Running validity check")
    if df.isnull().values.any():
        # Check for empty cells in the table
        log("Validity check: empty detected")
    if df.duplicated().any():
        # Check for duplicate lines in the table
        log("Validity check: duplicates detected")

if __name__ == "__main__":
    start = pd.Timestamp.now()

    contents = [] # stores all text content of all websites and sub-sites
    hrefs = [] # stores all links to sub-sites of all webistes

    # Reading the list of websites to scrape:
    if len(sys.argv) > 1:
        filename = str(sys.argv[1])
    else:
        filename = 'Websites.txt'

    # Reading the list of websites to be scraped for text:
    file = open(filename, 'r')
    websites = [line.strip("\n") for line in file.readlines()]
    file.close()
    log("Started scraping " + str(len(websites)) + " websites") # log message

    # Scraping every site in the list of websites:
    for site in websites:
        scrape(site, main=True)

    # Shuffling the list of sub-sites to avoid sending too many requests to the same site one after another:
    hrefs = list(set(hrefs))
    np.random.shuffle(hrefs)

    # Scraping every site in the list of sub-sites:
    for subsite in hrefs:
         scrape(subsite, main=False)

    # Converting the lists of data to a dataframe:
    df = pd.DataFrame(contents, columns=['Main', 'URL', 'Title', 'Text', 'Date']).sort_values(['Main', 'URL'])

    # Validating the data table:
    validate(df)

    # Storing the scraped content in the database:
    log("Writing scraped contents to database") # log message
    db(df, table=filename[:-4]) # table name corresponds to filename of the list of websites

    end = pd.Timestamp.now()
    log("Program ended after " + str(end - start))
