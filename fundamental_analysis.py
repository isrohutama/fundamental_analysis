import requests
from bs4 import BeautifulSoup as bs
import re
import json
import pandas as pd

import numpy as np 
from collections import OrderedDict

import yfinance as yf
from datetime import datetime, timedelta

class Fundamental_Analysis:
    def __init__(self, income_link, balance_link, cashflow_link, ticker_symbol, start_year):
        ## get data frame from links
        self.df_income_statement = self.mt_to_pd(income_link)
        self.df_balance_sheet = self.mt_to_pd(balance_link)
        self.df_cash_flow = self.mt_to_pd(cashflow_link)
        
        ## replace all empty cell with zero
        self.df_income_statement = self.df_income_statement.replace('', '0', regex=True)
        self.df_balance_sheet = self.df_balance_sheet.replace('', '0', regex=True)
        self.df_cash_flow = self.df_cash_flow.replace('', '0', regex=True)
        
        print('COLOUMNS:')
        print(self.df_income_statement.columns)
        print('INCOME STATEMENT:\n')
        print(self.df_income_statement['field_name'])
        print('BALANCE SHEET:\n')
        print(self.df_balance_sheet['field_name'])
        print('CASH FLOW:\n')
        print(self.df_cash_flow['field_name'])
        
        ## Rearrange the columns
        cols = list(self.df_income_statement.columns)
        new_cols = cols[1:][::-1]  # [::-1] is for reversing the order
        new_cols.insert(0, cols[0])
        #
        self.df_income_statement = self.df_income_statement[new_cols]
        self.df_balance_sheet = self.df_balance_sheet[new_cols]
        self.df_cash_flow = self.df_cash_flow[new_cols]
        
        ## Get the financial dates
        self.fin_dates = list(self.df_income_statement.columns[1:])
        
        ## Get the years and their corresponding dates
        self.fin_y2d = OrderedDict()
        for val in self.fin_dates:
            cur_year = val[:4]
            if not (cur_year in self.fin_y2d.keys()):
                self.fin_y2d[cur_year] = list()
            self.fin_y2d[cur_year].append(val)
        self.fin_years = [key for key in self.fin_y2d.keys()]
        
        ## self.fin_dates start index
        self.fin_dates_start = 0
        for idx, val in enumerate(self.fin_dates):
            if start_year in val:
                self.fin_dates_start = idx
                break
        
        ## self.fin_years start index
        self.fin_years_start = 0
        for key in self.fin_y2d.keys():
            if start_year == key:
                break
            self.fin_years_start += 1
        
        ## Get the share price history
        ticker_data = yf.Ticker(ticker_symbol)
        self.df_ticker = ticker_data.history(period='1d', start=self.fin_dates[0], end=datetime.now())
        #
        self.ticker_dates = [str(date)[:10] for date in self.df_ticker.index]
        self.ticker_prices = self.df_ticker['Close'].to_numpy(dtype=float).flatten()
        #
        self.ticker_start = 0
        for idx, val in enumerate(self.ticker_dates):
            if start_year in val:
                self.ticker_start = idx
                break
    
    def get_valuation(self):
        _, ttm_eps, eps_date, _, _ = self.get_fin_state_data('EPS - Earnings Per Share', 'sum', ttm=True, all_dates=True)
        _, ttm_net, _, _, _ = self.get_fin_state_data('Net Income', 'sum', ttm=True, all_dates=True)
        shareout, _, _, _, _ = self.get_fin_state_data('Shares Outstanding', 'last', ttm=False, all_dates=True)
        _, ttm_depamor, _, _, _ = self.get_fin_state_data('Total Depreciation And Amortization - Cash Flow', 'sum', ttm=True, all_dates=True)
        _, ttm_assliachg, _, _, _ = self.get_fin_state_data('Total Change In Assets/Liabilities', 'sum', ttm=True, all_dates=True)
        _, ttm_ppe, _, _, _ = self.get_fin_state_data('Net Change In Property, Plant, And Equipment', 'sum', ttm=True, all_dates=True)
        _, ttm_intax, _, _, _ = self.get_fin_state_data('Income Taxes', 'sum', ttm=True, all_dates=True)
        #
        ttm_fcf = ttm_net + ttm_depamor + ttm_assliachg + ttm_ppe + ttm_intax
        ttm_fcf_ps = [val / shareout[idx] for idx, val in enumerate(ttm_fcf.tolist())]
        #
        pe = np.zeros(len(self.ticker_dates))
        pfcf = np.zeros(len(self.ticker_dates))
        peg = np.zeros(len(self.ticker_dates))
        #
        eps_date_obj = [datetime.strptime(date, '%Y-%m-%d') for date in eps_date]
        ticker_date_obj = [datetime.strptime(date, '%Y-%m-%d') for date in self.ticker_dates]
        for idx1, date1 in enumerate(ticker_date_obj):
            last_eps = [ttm_eps[idx2] for idx2, date2_obj in enumerate(eps_date_obj) if (date2_obj <= date1)]
            last_fcf_ps = [ttm_fcf_ps[idx2] for idx2, date2_obj in enumerate(eps_date_obj) if (date2_obj <= date1)]
            last_net = [ttm_net[idx2] for idx2, date2_obj in enumerate(eps_date_obj) if (date2_obj <= date1)]
            if len(last_eps) >= 4:
                pe[idx1] = self.ticker_prices[idx1] / last_eps[-1]
                pfcf[idx1] = self.ticker_prices[idx1] / last_fcf_ps[-1]
            else:
                pe[idx1] = 0
                pfcf[idx1] = 0
            if len(last_eps) >= 8:
                growth_ttm1 = last_net[-1]
                growth_ttm2 = last_net[-5]
                growth_ttm = (growth_ttm1 - growth_ttm2) * 100 / growth_ttm2
                peg[idx1] = pe[idx1] / growth_ttm
            else:
                peg[idx1] = 0
        return pe[self.ticker_start:], pfcf[self.ticker_start:], peg[self.ticker_start:], ticker_date_obj[self.ticker_start:]
    
    '''
    ann = 'sum' or 'last'
    '''
    def get_fin_state_data(self, field_name, ann, ttm=False, all_dates=False):
        if field_name in self.df_income_statement['field_name'].values:
            df = self.df_income_statement
        elif field_name in self.df_balance_sheet['field_name'].values:
            df = self.df_balance_sheet
        elif field_name in self.df_cash_flow['field_name'].values:
            df = self.df_cash_flow
        else:
            return None, None, None, None, None
        #
        df_fn = df.loc[df['field_name'] == field_name]
        fn = df_fn.iloc[0, 1:].to_numpy(dtype=float).flatten()
        #
        if ttm:
            ttm_fn = np.zeros_like(fn)
            for idx in range(3, fn.shape[0]):
                ttm_fn[idx] = fn[idx] + fn[idx - 1] + fn[idx - 2] + fn[idx - 3]
        #
        ann_fn = np.zeros(len(self.fin_y2d))
        for idx, year in enumerate(self.fin_y2d):
            if ann == 'sum':
                for date in self.fin_y2d[year]:
                    ann_fn[idx] += float(df_fn[date].values[0])
            elif ann == 'last':
                ann_fn[idx] = float(df_fn[self.fin_y2d[year][-1]].values[0])
        #
        if all_dates:
            fin_dates_start = 0
            fin_years_start = 0
        else:
            fin_dates_start = self.fin_dates_start
            fin_years_start = self.fin_years_start
        if ttm:
            return fn[fin_dates_start:],\
                   ttm_fn[fin_dates_start:],\
                   self.fin_dates[fin_dates_start:],\
                   ann_fn[fin_years_start:],\
                   self.fin_years[fin_years_start:]
        else:
            return fn[fin_dates_start:],\
                   None,\
                   self.fin_dates[fin_dates_start:],\
                   ann_fn[fin_years_start:],\
                   self.fin_years[fin_years_start:]
            
    '''
    This method extracts then converts financial statements from macrotrends.com to data frame.
    Source: https://stackoverflow.com/questions/56417474/
            unable-to-retrieve-data-from-macro-trends-using-selenium-and-read-html-to-create
    '''
    @staticmethod
    def mt_to_pd(link):
        r = requests.get(link)
        p = re.compile(r' var originalData = (.*?);\r\n\r\n\r',re.DOTALL)
        data = json.loads(p.findall(r.text)[0])
        headers = list(data[0].keys())
        headers.remove('popup_icon')
        result = []

        for row in data:
            soup = bs(row['field_name'], features="lxml")
            field_name = soup.select_one('a, span').text
            fields = list(row.values())[2:]
            fields.insert(0, field_name)
            result.append(fields)

        pd.option_context('display.max_rows', None, 'display.max_columns', None)
        df = pd.DataFrame(result, columns = headers)

        return df