import argparse
import yaml
from fa import Fundamental_Analysis
import matplotlib.pyplot as plt 
import numpy as np

from datetime import datetime
import os

## Retrieve company name from user input
parser = argparse.ArgumentParser()
parser.add_argument('company_name', type=str, help='Company name as in fa_input_list.yaml')
args = parser.parse_args()

## Retrieve required input data from dcf_input_list.yaml
with open('fa_input_list.yaml', 'r') as f:
    data = yaml.safe_load(f)
income_statement = data[args.company_name]['income_statement']
balance_sheet = data[args.company_name]['balance_sheet']
cash_flow = data[args.company_name]['cash_flow']
ticker_symbol = data[args.company_name]['ticker_symbol']
start_year = data[args.company_name]['start_year']
output_prefix = data[args.company_name]['output_prefix']

## create output directory
dt_now = datetime.now()
out_dir = 'output/' + output_prefix + dt_now.strftime('_%Y%m%d_%H%M%S') + '/'
os.makedirs(out_dir)

## create string for md file
out_md_fn = out_dir + '0_fundamental_analysis.md'
out_md = '# ' + output_prefix + '  \n\n'

## function to write image fn in md file
def md_image(s, src, alt='image', width='1920'):
    s += '<img src=\'' + src + '\' alt=\'' + alt + '\' width=\'' + width + '\'/>  \n'
    return s

## function to draw a bar chart
def chart_bar(x, y, xlabel, ylabel, fn, out_md):
    '''
    x must be a list
    '''
    fig = plt.figure(dpi=150, figsize=(15, 5))
    ax = plt.axes()
    _X = np.arange(len(x))
    ax.bar(_X, y)
    ax.set_xticks(_X)
    ax.set_xticklabels(x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(fn)
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
    plt.close()
    out_md = md_image(out_md, fn + '.png')
    return out_md

## function to draw a bar chart with percentage values as secondary y-axis
def chart_bar_and_percentage(x, y, xlabel, ylabel, fn, out_md):
    '''
    x must be a list
    '''
    def val2perc(inp):
        return ((inp / y[0]) - 1.) * 100
    def perc2val(inp):
        return (1. + (inp / 100.)) * y[0]
    fig = plt.figure(dpi=150, figsize=(15, 5))
    ax1 = plt.axes()
    _X = np.arange(len(x))
    ax1.bar(_X, y)
    ax1.set_xticks(_X)
    ax1.set_xticklabels(x)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(fn)
    ax1.tick_params(axis='x', rotation=90)
    ax1.grid()
    if y[0] > 0.:
        ax2 = ax1.secondary_yaxis('right', functions=(val2perc, perc2val))
        ax2.set_ylabel('%')
    plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
    plt.close()
    out_md = md_image(out_md, fn + '.png')
    return out_md

## function to draw a chart with multiple bars
def chart_multi_bar(x, y, xlabel, ylabel, legend, fn, out_md):
    '''
    x must be a list
    y must be a list of arrays
    '''
    bar_num = len(y)
    bar_width = 0.9 / bar_num
    offset = np.arange(bar_num) * bar_width
    offset -= offset[-1] / 2.
    fig = plt.figure(dpi=150, figsize=(15, 5))
    ax = plt.axes()
    _X = np.arange(len(x))
    for i in range(bar_num):
        ax.bar(_X + offset[i], y[i], bar_width, label=legend[i])
    ax.set_xticks(_X)
    ax.set_xticklabels(x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(fn)
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    ax.legend()
    plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
    plt.close()
    out_md = md_image(out_md, fn + '.png')
    return out_md

## function to draw a chart with multiple lines
def chart_multi_plot(x, y, xlabel, ylabel, legend, fn, out_md):
    '''
    y must be a list of arrays
    '''
    plot_num = len(y)
    fig = plt.figure(dpi=150, figsize=(15, 5))
    ax = plt.axes()
    for i in range(plot_num):
        ax.plot(x, y[i], label=legend[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(fn)
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    ax.legend()
    plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
    plt.close()
    out_md = md_image(out_md, fn + '.png')
    return out_md

## function to draw a line chart with percentage values as secondary y-axis
def chart_plot_and_percentage(x, y, xlabel, ylabel, fn, out_md):
    '''
    x must be a list
    '''
    def val2perc(inp):
        return ((inp / y[0]) - 1.) * 100
    def perc2val(inp):
        return (1. + (inp / 100.)) * y[0]
    fig = plt.figure(dpi=150, figsize=(15, 5))
    ax1 = plt.axes()
    ax1.plot(x, y)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(fn)
    ax1.grid()
    if y[0] > 0.:
        ax2 = ax1.secondary_yaxis('right', functions=(val2perc, perc2val))
        ax2.set_ylabel('%')
    plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
    plt.close()
    out_md = md_image(out_md, fn + '.png')
    return out_md

## function to draw a line chart
def chart_plot(x, y, xlabel, ylabel, fn, out_md):
    '''
    x must be a list
    '''
    fig = plt.figure(dpi=150, figsize=(15, 5))
    ax1 = plt.axes()
    ax1.plot(x, y)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(fn)
    ax1.grid()
    plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
    plt.close()
    out_md = md_image(out_md, fn + '.png')
    return out_md

# extract information from financial statements and price data
fa = Fundamental_Analysis(income_statement, balance_sheet, cash_flow, ticker_symbol, start_year)

## Revenue
rev, ttm_rev, rev_date, ann_rev, rev_year = fa.get_fin_state_data('Revenue', 'sum', ttm=True)
rev_date_str = [date.strftime('%Y-%m-%d') for date in rev_date]
out_md = chart_bar_and_percentage(rev_date_str, rev, 'Date', 'Million USD', 'Revenue (Q)', out_md)
out_md = chart_bar_and_percentage(rev_date_str, ttm_rev, 'Date', 'Million USD', 'Revenue (TTM)', out_md)

## Gross Profit
gross, ttm_gross, _, ann_gross, _ = fa.get_fin_state_data('Gross Profit', 'sum', ttm=True)
out_md = chart_bar_and_percentage(rev_date_str, gross, 'Date', 'Million USD', 'Gross Profit (Q)', out_md)
out_md = chart_bar_and_percentage(rev_date_str, ttm_gross, 'Date', 'Million USD', 'Gross Profit (TTM)', out_md)

## Net Profit
net, ttm_net, _, ann_net, _ = fa.get_fin_state_data('Net Income', 'sum', ttm=True)
out_md = chart_bar_and_percentage(rev_date_str, net, 'Date', 'Million USD', 'Net Profit (Q)', out_md)
out_md = chart_bar_and_percentage(rev_date_str, ttm_net, 'Date', 'Million USD', 'Net Profit (TTM)', out_md)

## Gross Profit Margin, Net Profit Margin
gpm = gross / rev
npm = net / rev
ttm_gpm = ttm_gross / ttm_rev
ttm_npm = ttm_net / ttm_rev
ann_gpm = ann_gross / ann_rev
ann_npm = ann_net / ann_rev
#
out_md = chart_multi_bar(rev_date_str, [gpm*100, npm*100], 'Date', '%', ['GPM', 'NPM'], 'Gross Profit Margin, Net Profit Margin (Q)', out_md)
out_md = chart_multi_bar(rev_date_str, [ttm_gpm*100, ttm_npm*100], 'Date', '%', ['GPM', 'NPM'], 'Gross Profit Margin, Net Profit Margin (TTM)', out_md)

## Revenue Change
ttm_rev_chg = (ttm_rev - np.roll(ttm_rev, 1)) / np.roll(ttm_rev, 1)
ttm_rev_chg[0] = 0.
ttm_rev_chg4 = (ttm_rev - np.roll(ttm_rev, 4)) / np.roll(ttm_rev, 4)
ttm_rev_chg4[0] = 0.
ttm_rev_chg4[1] = 0.
ttm_rev_chg4[2] = 0.
ttm_rev_chg4[3] = 0.
ann_rev_chg = (ann_rev - np.roll(ann_rev, 1)) / np.roll(ann_rev, 1)
ann_rev_chg[0] = 0.
#
out_md = chart_bar(rev_date_str, ttm_rev_chg * 100, 'Date', '%', 'Revenue Change (TTM, Compared to Prev. Q)', out_md)
out_md = chart_bar(rev_date_str, ttm_rev_chg4 * 100, 'Date', '%', 'Revenue Change (TTM, Compared to Last Year)', out_md)
out_md = chart_bar(rev_year, ann_rev_chg * 100, 'Date', '%', 'Revenue Change (Ann)', out_md)

## Total Assets and Total Liabilities
ass, _, _, _, _ = fa.get_fin_state_data('Total Assets', 'last', ttm=False)
lia, _, _, _, _ = fa.get_fin_state_data('Total Liabilities', 'last', ttm=False)
#
out_md = chart_multi_bar(rev_date_str, [ass, lia], 'Date', 'Million USD', ['Total Assets', 'Total Liabilities'], 'Total Assets, Total Liabilities', out_md)

## Current Ratio
curass, _, _, _, _ = fa.get_fin_state_data('Total Current Assets', 'last', ttm=False)
curlia, _, _, _, _ = fa.get_fin_state_data('Total Current Liabilities', 'last', ttm=False)
#
cur_ratio = curass / curlia
#
out_md = chart_bar(rev_date_str, cur_ratio, 'Date', 'Ratio', 'Current Ratio', out_md)

## CFO, CFI, CFF
cfo, ttm_cfo, _, ann_cfo, _ = fa.get_fin_state_data('Cash Flow From Operating Activities', 'sum', ttm=True)
cfi, ttm_cfi, _, ann_cfi, _ = fa.get_fin_state_data('Cash Flow From Investing Activities', 'sum', ttm=True)
cff, ttm_cff, _, ann_cff, _ = fa.get_fin_state_data('Cash Flow From Financial Activities', 'sum', ttm=True)
#
out_md = chart_multi_plot(rev_date, [cfo, -cfi, -cff], 'Date', 'Million USD', ['CFO', '-CFI', '-CFF'], 'CFO, -CFI, -CFF (Q)', out_md)
out_md = chart_multi_plot(rev_date, [ttm_cfo, -ttm_cfi, -ttm_cff], 'Date', 'Million USD', ['CFO', '-CFI', '-CFF'], 'CFO, -CFI, -CFF (TTM)', out_md)
out_md = chart_multi_bar(rev_year, [ann_cfo, -ann_cfi,-ann_cff ], 'Year', 'Million USD', ['CFO', '-CFI', '-CFF'], 'CFO, -CFI, -CFF (Ann)', out_md)

## Free Cash Flow
cppe, ttm_cppe, _, ann_cppe, _ = fa.get_fin_state_data('Net Change In Property, Plant, And Equipment', 'sum', ttm=True)
#
fcf = cfo + cppe
ttm_fcf = ttm_cfo + ttm_cppe
ann_fcf = ann_cfo + ann_cppe
#
out_md = chart_bar_and_percentage(rev_date_str, fcf, 'Date', 'Million USD', 'Free Cash Flow (Q)', out_md)
out_md = chart_bar_and_percentage(rev_date_str, ttm_fcf, 'Date', 'Million USD', 'Free Cash Flow (TTM)', out_md)

## LT Debt / Asset
ltdebt, _, _, ann_debt, _ = fa.get_fin_state_data('Long Term Debt', 'last', ttm=True)
#
ltdebt_ass = ltdebt / ass
#
out_md = chart_bar(rev_date_str, ltdebt_ass, 'Date', 'Ratio', 'LT Debt - Asset Ratio', out_md)

## LT Debt / Equity
equ, _, _, ann_equ, _ = fa.get_fin_state_data('Share Holder Equity', 'last', ttm=True)
#
ltdebt_equ = ltdebt / equ
#
out_md = chart_bar(rev_date_str, ltdebt_equ, 'Date', 'Ratio', 'LT Debt - Equity Ratio', out_md)

## CFO / Net Profit
cfo_net = cfo / net
ttm_cfo_net = ttm_cfo / ttm_net
ann_cfo_net = ann_cfo / ann_net
#
out_md = chart_bar(rev_date_str, cfo_net, 'Date', 'Ratio', 'CFO - Net Profit Ratio (Q)', out_md)
out_md = chart_bar(rev_date_str, ttm_cfo_net, 'Date', 'Ratio', 'CFO - Net Profit Ratio (TTM)', out_md)

## ROE
ttm_roe = ttm_net / equ
ann_roe = ann_net / ann_equ
#
out_md = chart_bar(rev_date_str, ttm_roe * 100, 'Date', '%', 'Return on Equity (TTM)', out_md)

## EPS
eps, ttm_eps, _, ann_eps, _ = fa.get_fin_state_data('EPS - Earnings Per Share', 'sum', ttm=True)
out_md = chart_bar_and_percentage(rev_date_str, eps, 'Date', 'Million USD', 'Earning per Share (Q)', out_md)
out_md = chart_bar_and_percentage(rev_date_str, ttm_eps, 'Date', 'Million USD', 'Earning per Share (TTM)', out_md)

## Price related

price,pe, pfcf, prev, peg, ticker_date = fa.get_price_ratios()

print('price: ' + str(price[-1]))
print('P/E: ' + str(pe[-1]))
print('P/FCF: ' + str(pfcf[-1]))
print('P/S: ' + str(prev[-1]))
print('PEG: ' + str(peg[-1]))

out_md = chart_plot_and_percentage(ticker_date, price, 'Date', 'USD', 'Price', out_md)
out_md = chart_plot(ticker_date, pe, 'Date', 'USD', 'PE Ratio (TTM)', out_md)
out_md = chart_plot(ticker_date, prev, 'Date', 'USD', 'P - S Ratio (TTM)', out_md)
out_md = chart_plot(ticker_date, pfcf, 'Date', 'USD', 'P - FCF Ratio (TTM)', out_md)
out_md = chart_plot(ticker_date, peg, 'Date', 'PEG', 'PEG (TTM)', out_md)

## write string to file
f = open(out_md_fn, 'w')
f.write(out_md)
f.close()