import requests
from bs4 import BeautifulSoup
import time 
import datetime
# https://twitter.com/search?q=(%23aapl)%20until%3A2020-01-14%20since%3A2020-01-13

company_tickers = {"AAPL",
                   "MSFT",
                   "AMZN"}
# <div lang="en" dir="auto" class="css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0"><span dir="auto" class="css-901oao css-16my406 r-4qtqp9 r-ip8ujx r-sjv1od r-zw8f10 r-bnwqim r-h9hxbl"><div aria-label="Police cars revolving light" class="css-1dbjc4n r-xoduu5 r-1mlwlqe r-1d2f490 r-1udh08x r-u8s1d r-h9hxbl r-417010" style="height: 1.2em;"><div class="css-1dbjc4n r-1niwhzg r-vvn4in r-u6sd8q r-x3cy2q r-1p0dtai r-xoduu5 r-1pi2tsx r-1d2f490 r-u8s1d r-zchlnj r-ipm5af r-13qz1uu r-1wyyakw" style="background-image: url(&quot;https://abs-0.twimg.com/emoji/v2/svg/1f6a8.svg&quot;);"></div><img alt="Police cars revolving light" draggable="false" src="https://abs-0.twimg.com/emoji/v2/svg/1f6a8.svg" class="css-9pa8cd"></div></span><span class="css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0"> PRICE ALERT: </span><span class="r-18u37iz"><a href="/search?q=%24AAPL&amp;src=cashtag_click" dir="ltr" role="link" data-focusable="true" class="css-4rbku5 css-18t94o4 css-901oao css-16my406 r-1n1174f r-1loqt21 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0">$AAPL</a></span><span class="css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0"> reaches an all-time high, at $317.06 (+2.17%). </span><span class="css-901oao css-16my406 r-1qd0xha r-vw2c0b r-ad9z0x r-bcqeeo r-qvutc0"></span><span class="r-18u37iz"><a href="/hashtag/AAPL?src=hashtag_click" dir="ltr" role="link" data-focusable="true" class="css-4rbku5 css-18t94o4 css-901oao css-16my406 r-1n1174f r-1loqt21 r-1qd0xha r-vw2c0b r-ad9z0x r-bcqeeo r-qvutc0">#AAPL</a></span><span class="css-901oao css-16my406 r-1qd0xha r-vw2c0b r-ad9z0x r-bcqeeo r-qvutc0"></span><span class="css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0"> </span><span class="r-18u37iz"><a href="/hashtag/Apple?src=hashtag_click" dir="ltr" role="link" data-focusable="true" class="css-4rbku5 css-18t94o4 css-901oao css-16my406 r-1n1174f r-1loqt21 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0">#Apple</a></span></div>
def get_tweets(company_ticker, start_date, end_date):
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    print(start, end)
    seconds = 5
    url = 'https://twitter.com/search?q=(%23{})%20until%3A{}%20since%3A{}'.format(company_ticker, end, start)
    print(url)
    r1 = requests.get(url, time.sleep(seconds)) 
    coverpage = r1.text
    # print(coverpage)
    tweet_home = BeautifulSoup(coverpage, 'lxml')
    print(tweet_home)
    # tweet_list = tweet_home.find_all('div', class_='css-901oao r-jwli3a r-1qd0xha r-a023e6 r-16dba41 r-ad9z0x r-bcqeeo r-bnwqim r-qvutc0')
    tweet_list = tweet_home.find_all("li", {"data-item-type": "tweet"})

    print(tweet_list)
    articles_ret = []
    s = []
    for tweet in tweet_list:
        s.append(tweet.get_text())
        articles_ret.append('\n'.join(s))
        print(s)
        break
    return articles_ret

def main():
    article_map = {}
    for company_ticker in company_tickers:
        now = datetime.date.today()
        yesterday = now - datetime.timedelta(days=1)
        articles = get_tweets(company_ticker, yesterday, now)
        article_map[company_ticker] = articles

    for company_ticker in article_map:
        print(company_ticker, len(article_map[company_ticker]))

if __name__ == '__main__':
    main()
