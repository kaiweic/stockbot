from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from bs4 import BeautifulSoup
import time
import datetime
import re
# from rotate_proxies import get_proxies
import random
import os
import io

import configparser

config = configparser.ConfigParser()
config.read('settings.config')
config = config['DEFAULT']

chromedriver = config['chromedriver']
company_ticker = config['company_ticker']
DATA_DIR = config['DATA_DIR']
DATA_PATH = DATA_DIR + '{}_{}.tsv' # Should look like /d/stockbot_data/{}/{}_{}.tsv
alt = config.getboolean('alt')     # Should be true for eric since he's on linux

company_tags = {"AAPL": "AAPL",
                "MSFT": "MSFT",
                "BAC": "BankOfAmerica",
                "CMG": "Chipotle",
                "DAL": "Delta",
                "FB": "facebook",
                "GOOGL": "google",
                "JPM": "jpmorgan",
                "KO": "CocaCola",
                "LUV": "SouthwestAirlines",
                "MCD": "mcdonald",
                "PEP": "pepsi",
                "UAL": "ual",
                "V": "visa",
                "WFC": "wellsfargo",}

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--incognito')
chrome_options.add_argument('headless')
driver = None

def get_tweets(start_date, end_date, company_tag=company_ticker, alt=False):
    tweets = set()
    success = False
    link = None

    try:
        # r-30o5oe r-1niwhzg r-17gur6a r-1yadl64 r-deolkf r-homxoj r-poiln3 r-7cikom r-1ny4l3l r-1sp51qo r-1swcuj1 r-1dz5y72 r-1ttztb7 r-13qz1uu
        search_box = WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.CLASS_NAME, 'r-30o5oe')))
        time.sleep(random.uniform(1.5, 2.4))

        search_box.clear()
        time.sleep(0.4)
        
        # (#AMZN) until:2020-04-29 since:2020-04-28
        search_box.send_keys('(#{}) until:{} since:{}'.format(company_tag, end_date, start_date))
        search_box.submit()
        link = driver.current_url

        for i in range(10):
            y_off = driver.execute_script("return window.pageYOffset;")
            y_max = driver.execute_script("return document.body.scrollHeight;")
            driver.execute_script("window.scrollTo(0, {});".format((y_off + y_max) // 2))

            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, 'article')))
            time.sleep(random.uniform(1.4, 2.2))

            # css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0
            html = driver.page_source
            # articles = tweet_page.find_all('article')
            # for article in articles:
            #     text = article.find_all('div', class_="css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")

            if alt:
                tweet_page = BeautifulSoup(html, 'html.parser')
            else:
                tweet_page = BeautifulSoup(html, 'html5lib')

            # articles = tweet_page.find_all('div', class_="css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
            articles = tweet_page.find_all('div', class_="css-901oao r-hkyrab r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
            curr_tweets = set([re.sub(' +', ' ', article.get_text().replace('\n', ' ')) for article in articles])

            if curr_tweets.issubset(tweets):
                success = True
                break
            tweets = tweets | curr_tweets
        if i == 9:
            success = True
    except TimeoutException:
        print('Timing out, try again later')
    except NoSuchElementException:
        print("Can't find the element, try again later or maybe there aren't any tweets")
    except Exception as e:
        print(e)
        restart_driver()
    return list(tweets), success, link

# Add refreshing driver as a function, so it can be called in tweets_makeup.py
def restart_driver():
    global driver
    try:
        if driver is not None:
            print('restarting chromedriver')
            driver.quit()
    except Exception as e:
        print(e)
    driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)
    driver.get('https://www.twitter.com/')
    time.sleep(2)

def get_years(start_year=2010, end_year=2019, company_ticker=company_ticker, alt=False):
    global driver
    assert(start_year <= end_year)

    restart_driver()
    start_date = '01-01' # TODO CHANGE TO 01-01
    end_date = '01-01'   
    failed_year = {}

    path = DATA_DIR.format(company_ticker)
    print("Creating {} if not already created".format(path))
    if not os.path.exists(path):
        os.makedirs(path)

    company_tag = company_tags[company_ticker] if company_ticker in company_tags else company_ticker
    print("Searching up #{} for {} from {} to {} (inclusive)".format(company_tag, company_ticker, start_year, end_year))

    for year in range(start_year, end_year + 1):
        print('for year {}, from {} to {}'.format(year, str(year) + '-' + start_date, str(year + 1) + '-' + end_date))


        start_time = datetime.datetime.strptime(str(year) + '-' + start_date, '%Y-%m-%d')
        end_time = datetime.datetime.strptime(str(year + 1) + '-' + end_date, '%Y-%m-%d')

        dates = [start_time + datetime.timedelta(days=n) for n in range((end_time - start_time).days + 1)] # Accounts for leap years

        date_to_tweets = {}
        failed_links = {}

        try: 
            for i in range(1, len(dates)):
                start = dates[i - 1].strftime("%Y-%m-%d")
                end = dates[i].strftime("%Y-%m-%d")
                articles, success, link = get_tweets(start, end, company_tag=company_tag, alt=True)

                if not success:
                    failed_links[start] = link
                    print('failed')
                    print(link)

                print('got it for {} with {} results'.format(start, len(articles)))
                date_to_tweets[start] = articles
                time.sleep(3.5)

        except Exception as e: 
            print(e)

        year_path = DATA_PATH.format(company_ticker, company_ticker, str(year))
        print('writing year to {}\n'.format(year_path))
        with io.open(year_path, 'w', encoding='utf-8') as f:
            for date in date_to_tweets:
                tweets = date_to_tweets[date]
                for tweet in tweets:
                    f.write('{}\t{}\n'.format(date, tweet))

        if failed_links:
            failed_year[year] = failed_links

    if failed_year:
        print('writing failed dates to missing_tweets.txt')
        with io.open('./missing_tweets.txt', 'w', encoding='utf-8') as f:
            for year in failed_year:
                failed_links = failed_year[year]
                for date in failed_links:
                    link = failed_links[date]
                    f.write(date + " " + str(link) + "\n")
    else:
        print("There were no failed dates")
    driver.quit()


if __name__ == '__main__':
    get_years(start_year=2010, end_year=2019, alt=alt) # TODO Change to 2010 to 2019
    # driver.close()
