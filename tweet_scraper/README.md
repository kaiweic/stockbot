# Instructions for running tweet scraper by hash tag

0.  Initialize things in `settings.config`. It should look something like this 
```
[DEFAULT]
chromedriver = [CHROMEDRIVER PATH]
DATA_DIR = [WRITE PATH]
company_ticker = [COMPANY TICKER]
start_year = 2010
end_year = 2019
alt = [html5lib (false) vs. html.parser (true), html5lib by default]
cash = [cashtag, $ (true) vs. hashtag, # (false)]
```
TODO: Add start and end dates as config options, currently does Jan 1st 2010 to Jan 1st 2020 (exclusive)
1.  You run `selenium_tweets.py` after that
2.  It'll write failed dates and links to `missing_tweets.txt`. From there, you can run `makeup_tweets.py`
    and recovered tweets will be put in `recovered_tweets.txt` any failed dates back into `missing_tweets.txt`. 
    The end of `makeup_tweets.py` runs `rewrite_tweets.py` to write the recovered tweets to the files they 
    were missing from.
3.  Rerun `makeup_tweets.py` until there are no more missing dates or confirm that there were no tweets on that day. 
4.  When you have all the data, you can fill out and run `combine_data.py` to combine all the data, which will 
    write it to `./twitter_data/\[company_ticker\].tsv`, then from there, you can edit and run 
    `check_consecutive_tweets.py` to check whether there were any missing dates and confirm if the missing dates
    if those dates actually had no tweets with the links provided
