#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

contents = []
log = ""

clean = lambda txt: txt.replace("\t", "").replace("\n", "").replace("\r", "").replace("\xa0", " ")

def scrape_sub(href:str):
    global log
    try:
        response = requests.get(href, timeout=3)
    except:
        statement = "Error! Response timeout for " + href
        log += str(pd.Timestamp.today()) + " " + statement + "\n"
        print(statement)
    else:
        if int(response.status_code / 100) == 2:
            soup = BeautifulSoup(response.text, "lxml")
            try:
                text = clean(soup.get_text(" ", strip=True))
                timestamp = str(pd.Timestamp.today().date())
                contents.append((href, text, timestamp))
            except:
                statement = "Error! No text could be scraped for " + href
                log += str(pd.Timestamp.today()) + " " + statement + "\n"
                print(statement)
        else:
            statement = "Error! Response code " + str(response.status_code) + " for " + href
            log += str(pd.Timestamp.today()) + " " + statement + "\n"
            print(statement)

def scrape_main(url:str):
    global log
    try:
        response = requests.get(url, timeout=3)
    except:
        statement = "Error! Response timeout for " + url
        log += str(pd.Timestamp.today()) + " " + statement + "\n"
        print(statement)
    else:
        if int(response.status_code / 100) == 2:
            soup = BeautifulSoup(response.text, "lxml")
            try:
                text = clean(soup.get_text(" ", strip=True))
            except:
                statement = "Error! No text could be scraped for " + url
                log += str(pd.Timestamp.today()) + " " + statement + "\n"
                print(statement)
            timestamp = str(pd.Timestamp.today().date())
            contents.append((url, text, timestamp))

            list_of_href = []
            for a in soup.findAll('a', href=True):
                href = a['href']
                if href.startswith("/") and len(href) > 1:
                    list_of_href.append(href)
            list_of_href = list(set(list_of_href))
            list_of_href.sort()

            for h in list_of_href:
                scrape_sub(re.sub("/$", "", url) + h)
        else:
            statement = "Error! Response code " + str(response.status_code) + " for " + href
            log += str(pd.Timestamp.today()) + " " + statement + "\n"
            print(statement)

if __name__ == "__main__":
    file = open("websites.txt", 'r')
    websites = file.readlines()
    statement = "Started scraping " + str(len(websites)) + " websites"
    log += str(pd.Timestamp.today()) + " " + statement + "\n"
    print(statement)
    for site in websites:
        scrape_main(site.strip("\n"))
    file.close()
    statement = "Finished scraping " + str(len(contents)) + " subsites"
    log += str(pd.Timestamp.today()) + " " + statement + "\n"
    print(statement)
    file = open("log.txt", 'w+')
    file.write(log)
    file.close()
    df = pd.DataFrame(contents, columns=['url', 'text', 'timestamp'])
    df.to_csv("contents.csv", index=False, sep="\t")
