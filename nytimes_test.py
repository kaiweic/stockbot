import requests
from bs4 import BeautifulSoup
import time 

company_name = "apple-incorporated"
r1 = requests.get('https://www.nytimes.com/topic/company/' + company_name) 
coverpage = r1.content
article_home = BeautifulSoup(coverpage, 'html5lib')
article_list = article_home.find_all('div', class_='css-1l4spti')
time.sleep(6)
for article in article_list:
    children = article.findChildren('a', href=True, recursive=False)
    child_link = children[0]['href']
    print(child_link)
    article_request = requests.get('https://www.nytimes.com/' + child_link)
    article_content = BeautifulSoup(article_request.content, 'html5lib')
    paragraph_list = article_content.find_all('p', class_='css-exrw3m evys1bk0')
    for paragraph in paragraph_list:
        print(paragraph.get_text())
    print()
    time.sleep(6)
