"""
Evaluate the performance of hedge fund strategies
"""

import yfinance as yf
import pandas as pd, numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

# set the style of the plots
sns.set_theme(style='ticks', palette='husl', font_scale=1.2)


class HedgeFund:
    def __init__(self, name, directory, n_weeks=13):
        self.name = name
        self.directory = directory
        self.quarters = 0
        self.n_weeks = n_weeks
        self.positions = pd.DataFrame()
        self.price = pd.DataFrame()
        self.price_change = pd.DataFrame()
        self.sp500_change = pd.DataFrame()
        self.comparison = pd.DataFrame()
        self.portfolios = {}
        self.performance = {}


    def get_positions(self):
        """
        Get the positions data for the hedge fund by reading CSV files 
        derived from the 13F filings
        Assumes the CSV files are named in the format 'Deerfield_YYYY-MM-DD.csv'
        where the date is the filing date of the 13F form
        """
        # get all files in the directory that end with .csv
        filenames = [filename for filename in os.listdir(self.directory) if filename.endswith('.csv')]
        self.quarters = len(filenames) - 1

        ls = []
        for filename in filenames:
            positions = pd.read_csv(os.path.join(self.directory, filename))
            positions.insert(0, 'date', filename.split('_')[1].split('.')[0])
            ls.append(positions)

        positions = pd.concat(ls, axis=0, ignore_index=True).rename(columns={'Sym': 'ticker', 'Cl': 'class'})
        # convert all columns to lowercase
        positions.columns = positions.columns.str.lower()
        positions = positions.dropna(subset='ticker').sort_values(['date', 'ticker'])
        # drop rows where Cl contains an expiration date
        positions = positions[~positions['class'].str.contains('EXP|UNIT', regex=True, case=False)]
        positions['date'] = pd.to_datetime(positions['date'])
        # the potential sell date is 13 weeks after the buy date
        positions['sell_date'] = positions['date'] + pd.DateOffset(weeks=self.n_weeks)

        # determine the change in the number of shares held by Deerfield Management over time
        positions['shares'] = positions['shares'].str.replace(',', '').astype(float)
        positions['change'] = positions.groupby('ticker')['shares'].diff()
        # if there was no previous position, assume the change is the same as the current position
        positions['change'] = positions['change'].fillna(positions['shares'])
        # change must still be nan if the rows are from the earliest date
        positions.loc[positions['date'] == positions['date'].min(), 'change'] = np.nan
        
        positions['value'] = positions['value ($000)'].str.replace(',', '').astype(float) * 1000

        # indicate the quarter for each row
        positions['quarter'] = 'q' + positions['date'].dt.quarter.astype(str) + '_' + positions['date'].dt.year.astype(str)
        self.positions = positions


    def get_prices(self):
        """
        Get the price data for the positions that increased in value
        """
        # get price data for the positions that increased
        buy_tickers = list(self.positions.query("change > 0")['ticker'].unique())
        # position may have increased because of a stock split, rather than a purchase
        # --> need to remove these false positives

        # cannot determine share changes from the earliest date. Use the next date instead
        start = self.positions['date'].unique()[1].astype(str)[:10]
        end = self.positions['sell_date'].max().strftime('%Y-%m-%d')
        price = yf.download(buy_tickers, start=start, end=end, interval='1wk')['Adj Close']
        self.price = price

        # calculate the percentage change in price over the past n_weeks
        price_change = price.pct_change(self.n_weeks) * 100
        # melt the price data to make it easier to merge with the positions data
        price_change = price_change.reset_index().melt(
            id_vars='Date', var_name='ticker', value_name='price_change').rename(columns={'Date': 'sell_date'})
        price_change['sell_date'] = pd.to_datetime(price_change['sell_date'])
        price_change.sort_values(['sell_date', 'ticker'], inplace=True)
        self.price_change = price_change

        # get data for VOO
        sp500 = yf.download('VOO', start=start, end=end, interval='1wk')['Adj Close']
        sp500_change = sp500.pct_change(self.n_weeks) * 100
        sp500_change = sp500_change.reset_index().rename(columns={'Adj Close': 'VOO', 'Date': 'sell_date'})
        sp500_change['sell_date'] = pd.to_datetime(sp500_change['sell_date'])
        self.sp500_change = sp500_change


    def create_portfolio(self, approach='simple'):
        """
        Create a portfolio based on the positions data
        Currently, the portfolio is a simple strategy that buys (and holds) stocks with
        the same weighting as the hedge fund. I want to add more complex strategies in the future.
        For example, a strategy that only buys stocks that have increased in value over the past n weeks.
        It would be interesting to compare how different strategies perform over time.
        """
        # merge the positions data with the price data. the inner merge will drop rows where the price data is missing
        portfolio = pd.merge_asof(self.positions, self.price_change, on='sell_date', 
                                  by='ticker', direction='nearest')

        # calculate the weight of each position as a proportion of the total value per quarter
        portfolio['weight'] = portfolio.groupby('quarter')['value'].transform(lambda x: x / x.sum())

        # calculate performance as the weighted sum of the returns
        portfolio['performance'] = portfolio['price_change'] * portfolio['weight']

        performance = portfolio.groupby(
            ['quarter','sell_date'])['performance'].sum().reset_index().sort_values('sell_date').reset_index(drop=True)
        performance.loc[0, 'performance'] = np.nan

        self.portfolios[approach] = portfolio
        self.performance[approach] = performance


    def calculate_returns(self):
        """
        Calculate the returns of the hedge fund strategy compared to the S&P 500
        Uses a simple stragegy for the portfolio weighting
        """
        # if there are no portfolios, create the simple portfolio
        if not self.performance:
            self.create_portfolio()        
        
        if len(self.performance) > 1:
            # combine the performance of the different portfolios into one dataframe by merging on the sell_date
            ls = [df.rename(columns={'performance':approach}).drop(columns='quarter').set_index('sell_date') 
                  for approach, df in self.performance.items()]
            performance = pd.concat(ls, axis=1).reset_index()
        else:
            performance = list(self.performance.values())[0]

        # compare the performance to the S&P 500
        comparison = pd.merge_asof(performance, self.sp500_change, 
                                   on='sell_date', direction='nearest')
        if 'performance' in comparison.columns:
            comparison['outperformance'] = comparison['performance'] - comparison['VOO']
        comparison = comparison.round(2).sort_values(by='sell_date')
        self.comparison = comparison
    

    def evaluate(self):
        """
        Evaluate the performance of the hedge fund strategy

        Returns:
        comparison (pd.DataFrame): the performance of the hedge fund strategy compared to the S&P 500
        """
        self.get_positions()
        self.get_prices()
        self.calculate_returns()
        return self.comparison


    def plot_growth(self, principal=10_000):
        """
        Plot the growth of the investment over time
        """
        # top row will be the initial investment
        growth = self.comparison.fillna(0)
        # identify the investment approaches used
        approaches = growth.columns[~growth.columns.isin(['quarter', 'sell_date','outperformance'])]
        growth[approaches] = growth[approaches].div(100) + 1
        growth[approaches] = growth[approaches].cumprod() * principal
        growth = growth.round(2)

        growth.plot(x='sell_date', y=[approaches])
        plt.ylabel(f'Value of ${principal:,.0f} investment')

    
if __name__ == '__main__':
    # get the current working directory
    top_directory = os.getcwd()

    # create a HedgeFund object for Deerfield Management
    deerfield = HedgeFund('Deerfield', os.path.join(top_directory, 'deerfield'))
    deerfield.evaluate()
    deerfield.plot_growth()
    plt.show()

    # # create a HedgeFund object for Point72 Asset Management
    # point72 = HedgeFund('Point72', os.path.join(top_directory, 'point72'))
    # point72.evaluate()
    # point72.plot_growth()
    # plt.show()