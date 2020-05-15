import selenium_tweets 

company_ticker = 'MSFT'
DATA_DIR = './stockbot_data/{}/'

company_tags = selenium_tweets.company_tags

selenium_tweets.company_ticker = company_ticker
selenium_tweets.DATA_DIR = DATA_DIR
selenium_tweets.DATA_PATH = selenium_tweets.DATA_DIR + '{}_{}.tsv'

DATA_PATH = selenium_tweets.DATA_PATH

# selenium_tweets.chromedriver = "/c/Users/andre/Documents/drivers/chromedriver-81/chromedriver.exe"
selenium_tweets.chromedriver = "./chromedriver"

company_tag = company_tags[company_ticker] if company_ticker in company_tags else company_ticker

def get_tweets(start, end):
    print(company_ticker, company_tag)
    selenium_tweets.get_tweets(start, end, company_tag)

def main():
    selenium_tweets.get_years(start_year=2010, end_year=2013, company_ticker=company_ticker) # TODO Change end_year to 2019

if __name__ == '__main__':
    main()
