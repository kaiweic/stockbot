from selenium import webdriver
from rotate_proxies import get_proxies
import random
from bs4 import BeautifulSoup
import time
import re
import datetime

def 

chromedriver = "/c/Users/andre/Documents/drivers/chromedriver-81/chromedriver.exe"
# https://www.labnol.org/internet/setup-proxy-server/12890/
# proxies = get_proxies()
# proxy = random.sample((proxies), 1)[0]

tweets = set()

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--incognito')
chrome_options.add_argument('headless')
# chrome_options.add_argument('--proxy-server={}'.format(proxy))

driver = webdriver.Chrome(chromedriver, chrome_options=chrome_options)
driver.get('http://www.twitter.com/');
time.sleep(random.randint(2, 4))

# r-30o5oe r-1niwhzg r-17gur6a r-1yadl64 r-deolkf r-homxoj r-poiln3 r-7cikom r-1ny4l3l r-1sp51qo r-1swcuj1 r-1dz5y72 r-1ttztb7 r-13qz1uu
search_box = driver.find_element_by_class_name('r-30o5oe')

# (#AMZN) until:2020-04-29 since:2020-04-28
company_ticker = 'AMZN'

now = datetime.date.today()
yesterday = now - datetime.timedelta(days=1)

end_date, start_date = now, yesterday

search_box.send_keys('(#{}) until:{} since:{}'.format(company_ticker, end_date, start_date))
search_box.submit()
print(driver.current_url)
# for i in range(2):
time.sleep(2)

for i in range(10):
    y_off = driver.execute_script("return window.pageYOffset;")
    y_max = driver.execute_script("return document.body.scrollHeight;")
    driver.execute_script("window.scrollTo(0, {});".format((y_off + y_max) // 2))
    time.sleep(random.uniform(1.1, 1.5))

    # css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0
    html = driver.page_source
    # articles = tweet_page.find_all('article')
    # for article in articles:

    #     text = article.find_all('div', class_="css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
    tweet_page = BeautifulSoup(html, 'html5lib')
    # articles = tweet_page.find_all('div', class_="css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
    articles = tweet_page.find_all('div', class_="css-901oao r-hkyrab r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0")
    curr_tweets = set([re.sub(' +', ' ', article.get_text().replace('\n', ' ')) for article in articles])
    if curr_tweets.issubset(tweets):
        break
    tweets = tweets | curr_tweets
driver.quit()
for tweet in tweets: 
    print('AMZN', tweet)

print(len(tweets))