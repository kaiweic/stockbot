import argparse
import datetime
import requests
import pytz
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Price data scraper")
    parser.add_argument('--tickers', nargs='+', help='Ticker of interest')
    parser.add_argument('--start_date', help="Start date. Format: YYYY-MM-DD")
    parser.add_argument('--end_date', help="End date. Format: YYYY-MM-DD")
    args = parser.parse_args()

    try:
        datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    except:
        print("Failed to parse the start_date or end_date. Please make sure data are in the correct format")

    ti = TickerInfo()
    ti.get_data(args.tickers, args.start_date, args.end_date)


class TickerInfo:
    URL = "https://api.polygon.io/"
    API_KEY = "KEY_HERE"


    def get_data(self, tickers, start_date, end_date):
        total = pd.DataFrame()
        for ticker in tickers:
                endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
                r = requests.get(TickerInfo.URL + endpoint,
                                 params={'apiKey': TickerInfo.API_KEY, 'unadjusted': 'false'})
                data = r.json()['results']
                result = pd.DataFrame()
                for day_context in data:
                    idx = len(result)
                    result.loc[idx, "date"] = datetime.datetime.fromtimestamp(float(day_context['t']) / 1000, tz=pytz.timezone('US/Eastern')).replace(tzinfo=None).strftime("%Y-%m-%d")
                    result.loc[idx, "open"] = day_context['o']
                    result.loc[idx, "high"] = day_context['h']
                    result.loc[idx, "low"] = day_context['l']
                    result.loc[idx, "close"] = day_context['c']
                    result.loc[idx, "volume"] = float(day_context['v']) / 1000000
                result.to_csv(f"{ticker}_{start_date}_{end_date}.csv")




if __name__ == "__main__":
    main()

