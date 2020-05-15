import selenium_tweets 

company_ticker = 'MSFT'
DATA_DIR = './stockbots_data/{}/'

selenium_tweets.company_ticker = company_ticker
selenium_tweets.DATA_DIR = DATA_DIR
selenium_tweets.DATA_PATH = selenium_tweets.DATA_DIR + '{}_{}.tsv'

DATA_PATH = selenium_tweets.DATA_PATH

#chromedriver = "/c/Users/andre/Documents/drivers/chromedriver-81/chromedriver.exe"
selenium_tweets.chromedriver = "./chromedriver"

company_tag = selenium_tweets.company_tags[company_ticker] if company_ticker in selenium_tweets.company_tags else company_ticker

def get_tweets(start, end):
    selenium_tweets.get_tweets(start, end, company_tag)

def main():
    selenium_tweets.get_years(start_year=2010, end_year=2019)

if __name__ == '__main__':
    main()
    driver.close()
