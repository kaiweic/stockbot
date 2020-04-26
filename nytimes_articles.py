import requests
import time
import csv
import json 

# Following this website: 
# https://github.com/Beehamer/cs229stockprediction/blob/master/src/NYtimesScraper.py

# Setting up API Key access
with open('API_KEYS.txt') as f:
    keys = json.load(f)
api_key = keys['NYTimes']

def parse_articles(articles):
    news = []
    i = 0
    try:
        for article in articles['response']['docs']:
            print(i, article)
            i += 1
            dic = {}
            dic['id'] = article['_id']
            dic['headline'] = article['headline']['main'].encode('utf8')
            dic['date'] = article['pub_date'][0:10]
            if article['snippet'] is not None:
                dic['snippet'] = article['snippet'].encode('utf8')
                dic['source'] = article['source']
                dic['type'] = article['type_of_material']
                dic['url'] = article['web_url']
                dic['word_count'] = article['word_count']

                locations = []
                for x in range(0, len(article['keywords'])):
                    if 'glocations' in article['keywords'][x]['name']:
                        locations.append(article['keywords'][x]['value'])
                        dic['locations'] = locations

                        subjects = []
                        for x in range(0, len(article['keywords'])):
                            if 'subject' in article['keywords'][x]['name']:
                                subjects.append(article['keywords'][x]['value'])
                                dic['subjects'] = subjects
                                news.append(dic)
    except Exception as e:
        print(e)
    return news 

def get_articles(date, query):
    all_articles = []
    for i in range(3):
        payload = {"q": query,
                   "fq": {'source':['Reuters','AP', 'The New York Times']},
                   "begin_date": date,
                   "end_date": date,
                   "sort" : 'newest',
                   "news_desk" : 'business',
                   "subject" : 'business',
                   "glocations" : 'U.S.',
                   "page": str(i),
                   "api-key": api_key}
        try:
            articles = requests.get("https://api.nytimes.com/svc/search/v2/articlesearch.json?", params=payload)
            articles = json.loads(articles.text)
            articles = parse_articles(articles)
            time.sleep(6)
            all_articles += articles 
        except Exception as e:
            print(e)
    return all_articles


from datetime import date

def main():
    companies = ['AAPL']
    today = ''.join(str(date.today()).split('-'))
    for company in companies:
        query = company
        company_news = get_articles(today, company)
        print(company_news)

if __name__ == '__main__':
    main()