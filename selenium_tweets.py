from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from bs4 import BeautifulSoup
import time
import datetime
import re

company_tickers = {"AAPL",
                   "MSFT",
                   "AMZN"}

chromedriver = "/c/Users/andre/Documents/drivers/chromedriver-81/chromedriver.exe"
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--incognito')
chrome_options.add_argument('headless')

def get_tweets(company_ticker, start_date, end_date):
    tweets = []
    driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)
    try:
        driver.get('http://www.twitter.com/');
        time.sleep(5)
        
        # r-30o5oe r-1niwhzg r-17gur6a r-1yadl64 r-deolkf r-homxoj r-poiln3 r-7cikom r-1ny4l3l r-1sp51qo r-1swcuj1 r-1dz5y72 r-1ttztb7 r-13qz1uu
        search_box = driver.find_element_by_class_name('r-30o5oe')
        
        # (#AMZN) until:2020-04-29 since:2020-04-28
        search_box.send_keys('(#{}) until:{} since:{}'.format(company_ticker, end_date, start_date))
        search_box.submit()
        print(driver.current_url)
        time.sleep(5)

        # css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0
        html = driver.page_source
        # articles = tweet_page.find_all('article')
        # for article in articles:
        #     text = article.find_all('div', class_="css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
        tweet_page = BeautifulSoup(html, 'html5lib')
        # articles = tweet_page.find_all('div', class_="css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
        articles = tweet_page.find_all('div', class_="css-901oao r-hkyrab r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
        driver.quit()
        for article in articles:
            tweets.append(re.sub(' +', ' ', article.get_text().replace('\n', ' ')))
    except Exception as e:
        print(e)
        driver.quit()
    return tweets


def main():
    article_map = {}
    for company_ticker in company_tickers:
        now = datetime.date.today()
        yesterday = now - datetime.timedelta(days=1)
        articles = get_tweets(company_ticker, yesterday, now)
        article_map[company_ticker] = articles

    for company_ticker in article_map:
        for article in article_map[company_ticker]:
            print(company_ticker, article)
if __name__ == '__main__':
    main()
