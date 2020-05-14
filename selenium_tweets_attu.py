from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
#from chromedriver_py import binary_path
from bs4 import BeautifulSoup
import time
import datetime
import re
# from rotate_proxies import get_proxies
import random

company_tags = {"AAPL": "AAPL",
                "MSFT": "MSFT",
                "BAC": "BankOfAmerica",
                "CMG": "Chipotle",
                "DAL": "Delta",
                "FB": "facebook",
                "GOOGL": "GOOGL",
                "JPM": "jpmorgan",
                "KO": "CocaCola",
                "LUV": "SouthwestAirlines",
                "MCD": "mcdonald",
                "PEP": "pepsi",
                "UAL": "ual",
                "V": "visa",
                "WFC": "wellsfargo",}

# proxies = get_proxies()
# print(proxies)

#chromedriver = "/c/Users/andre/Documents/drivers/chromedriver-81/chromedriver.exe"
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--incognito')
chrome_options.add_argument('headless')
driver = webdriver.Chrome("./chromedriver", chrome_options=chrome_options)
driver.get('https://www.twitter.com/');

def get_tweets(company_ticker, company_tag, start_date, end_date):
    tweets = set()
    # proxy = random.sample((proxies), 1)[0]
    # print('using proxy {}'.format(proxy))

    # chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--incognito')
    # chrome_options.add_argument('headless')
    # # chrome_options.add_argument('--proxy-server={}'.format(proxy))
    # driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)
    success = False
    link = None

    try:
        # r-30o5oe r-1niwhzg r-17gur6a r-1yadl64 r-deolkf r-homxoj r-poiln3 r-7cikom r-1ny4l3l r-1sp51qo r-1swcuj1 r-1dz5y72 r-1ttztb7 r-13qz1uu
        search_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'r-30o5oe')))
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
            tweet_page = BeautifulSoup(html, 'html.parser')
            # articles = tweet_page.find_all('div', class_="css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
            articles = tweet_page.find_all('div', class_="css-901oao r-hkyrab r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
            curr_tweets = set([re.sub(' +', ' ', article.get_text().replace('\n', ' ')) for article in articles])

            if curr_tweets.issubset(tweets):
                success = True
                break
            tweets = tweets | curr_tweets
    except Exception as e:
        print(e)
    return list(tweets), success, link


def main():
    global driver 
    company_ticker = 'MSFT'
    year = 2010
    month_map = []
    start_date = '12-31'
    end_date = '12-31'
    
    print('for year {}, from {} to {}'.format(year, str(year - 1) + '-' + start_date, str(year) + '-' + end_date))
    company_tag = company_tags[company_ticker]

    start_time = datetime.datetime.strptime(str(year - 1) + '-' + start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(str(year) + '-' + end_date, '%Y-%m-%d')

    dates = [start_time + datetime.timedelta(days=n) for n in range((end_time - start_time).days + 1)]

    date_to_tweets = {}

    failed_links = {}
    try: 
        for i in range(1, len(dates)):
            start = dates[i - 1].strftime("%Y-%m-%d")
            end = dates[i].strftime("%Y-%m-%d")
            articles, success, link = get_tweets(company_ticker, company_tag, start, end)
            if not success:
                failed_links[end] = link
                print('failed')
                print(link)
            print('got it for {} with {} results'.format(end, len(articles)))
            date_to_tweets[end] = articles
            time.sleep(3.2)
            if i % 10 == 0:
                driver = webdriver.Chrome("./chromedriver", chrome_options=chrome_options)
                driver.get('https://www.twitter.com/');
                time.sleep(2)
    except Exception as e: 
        print(e)

    with open('./stockbots_data/{}/{}_{}.tsv'.format(company_ticker, company_ticker, year), 'w') as f:
        for date in date_to_tweets:
            tweets = date_to_tweets[date]
            for tweet in tweets:
                f.write('{}\t{}\n'.format(date, tweet))

    if failed_links:
        print('failed dates')
        for date in failed_links:
            link = failed_links[date]
            print(date, link)

if __name__ == '__main__':
    main()
    driver.close()
