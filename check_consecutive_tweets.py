import datetime
import selenium_tweets
import os
import io

company_ticker = selenium_tweets.company_ticker

PATH = selenium_tweets.DATA_PATH

company_tag = selenium_tweets.company_tags[company_ticker] if company_ticker in selenium_tweets.company_tags else company_ticker

start_year = 2010
end_year = 2019
all_days = set()
for year in range(start_year, end_year + 1):
    start_date = datetime.datetime.strptime("{}-01-01".format(year), '%Y-%m-%d')
    end_date = datetime.datetime.strptime("{}-01-01".format(int(year) + 1), '%Y-%m-%d')
    days_in_year = (end_date - start_date).days
    curr_days = set([(start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_in_year)]) # accounts for leap years

    curr_path = PATH.format(company_ticker, company_ticker, year)
    if not os.path.exists(curr_path):
        print("Data for {} doesn't exist, maybe check that later".format(year))
        continue

    date = None
    with io.open(curr_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                date, _ = line.split('\t', 1)
            except Exception as e:
                print(e)
                if date:
                    print(date)
                print(line)
            if date in curr_days: curr_days.remove(date)
    all_days |= curr_days

missing_dates = sorted(all_days)
print()
print("check the following dates")
print(missing_dates)
print()
print('list for missing_tweets.txt')
for date in missing_dates:
    tomorrow = (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    print('{} https://twitter.com/search?q=(%23{})%20until%3A{}%20since%3A{}&src=typed_query'.format(date, company_tag, tomorrow, date))
