import requests
from bs4 import BeautifulSoup
import time 
import datetime

companies_map = {"AAPL": "apple-incorporated",
                 "MSFT": "microsoft-corporation",
                 "AMZN": "amazoncom-inc",}

def get_articles(company_name):
    r1 = requests.get('https://www.nytimes.com/topic/company/' + company_name) 
    coverpage = r1.content
    article_home = BeautifulSoup(coverpage, 'html5lib')
    article_list = article_home.find_all('div', class_='css-1l4spti')
    time.sleep(6)
    articles_ret = []
    for article in article_list:
        children = article.findChildren('a', href=True, recursive=False)
        child_link = children[0]['href']
        print(child_link)
        article_request = requests.get('https://www.nytimes.com/' + child_link)
        article_content = BeautifulSoup(article_request.content, 'html5lib')
        date = datetime.datetime.strptime(article_content.find_all('time')[0]['datetime'][:-5], '%Y-%m-%dT%H:%M:%S-')
        if date.date() < datetime.date.today() - datetime.timedelta(days=7):
            break
        paragraph_list = article_content.find_all('p', class_='css-exrw3m evys1bk0')
        s = []
        for paragraph in paragraph_list:
            s.append(paragraph.get_text())
        articles_ret.append('\n'.join(s))
        time.sleep(6)
    return articles_ret

def main():
    article_map = {}
    for company_ticker in companies_map:
        company_name = companies_map[company_ticker]
        print(company_name)
        articles = get_articles(company_name)
        article_map[company_name] = articles

    for company in article_map:
        print(company, len(article_map[company]))

if __name__ == '__main__':
    main()
