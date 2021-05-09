##########################################
########## REQUIRED USER INPUTS ##########
##########################################

## links from macrotrends
income_statement = 'https://www.macrotrends.net/stocks/charts/GOOGL/alphabet/income-statement?freq=Q'
balance_sheet = 'https://www.macrotrends.net/stocks/charts/GOOGL/alphabet/balance-sheet?freq=Q'
cash_flow = 'https://www.macrotrends.net/stocks/charts/GOOGL/alphabet/cash-flow-statement?freq=Q'

## needed information to retrieve data from yfinance
ticker_symbol = 'GOOGL'
start_year = '2011'

output_prefix = 'alphabet'

##########################################
############ PROCESSING CODE #############
##########################################

from fundamental_analysis import Fundamental_Analysis
import matplotlib.pyplot as plt 
import numpy as np

from datetime import datetime
import os

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

# extract information from financial statements and price data
fa = Fundamental_Analysis(income_statement, balance_sheet, cash_flow, ticker_symbol, start_year)

## Revenue, Gross Profit, Net Profit
rev, ttm_rev, rev_date, ann_rev, rev_year = fa.get_fin_state_data('Revenue', 'sum', ttm=True)
gross, ttm_gross, _, ann_gross, _ = fa.get_fin_state_data('Gross Profit', 'sum', ttm=True)
net, ttm_net, _, ann_net, _ = fa.get_fin_state_data('Net Income', 'sum', ttm=True)
#
fn = 'Revenue, Gross Profit, Net Profit (Q)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.plot(rev_date, rev, label='Revenue')
ax.plot(rev_date, gross, label='Gross Profit')
ax.plot(rev_date, net, label='Net Profit')
ax.set_xlabel('Date')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Revenue, Gross Profit, Net Profit (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.plot(rev_date, ttm_rev, label='Revenue')
ax.plot(rev_date, ttm_gross, label='Gross Profit')
ax.plot(rev_date, ttm_net, label='Net Profit')
ax.set_xlabel('Date')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Revenue, Gross Profit, Net Profit (Ann)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_year, ann_rev, label='Revenue')
ax.bar(rev_year, ann_gross, label='Gross Profit')
ax.bar(rev_year, ann_net, label='Net Profit')
ax.set_xlabel('Date')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## Gross Profit Margin, Net Profit Margin
gpm = gross / rev
npm = net / rev
ttm_gpm = ttm_gross / ttm_rev
ttm_npm = ttm_net / ttm_rev
ann_gpm = ann_gross / ann_rev
ann_npm = ann_net / ann_rev
#
fn = 'Gross Profit Margin, Net Profit Margin (Q)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, gpm*100, label='GPM')
ax.bar(rev_date, npm*100, label='NPM')
ax.set_xlabel('Date')
ax.set_ylabel('%')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Gross Profit Margin, Net Profit Margin (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, ttm_gpm*100, label='GPM')
ax.bar(rev_date, ttm_npm*100, label='NPM')
ax.set_xlabel('Date')
ax.set_ylabel('%')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Gross Profit Margin, Net Profit Margin (Ann)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_year, ann_gpm*100, label='GPM')
ax.bar(rev_year, ann_npm*100, label='NPM')
ax.set_xlabel('Date')
ax.set_ylabel('%')
ax.set_title(fn)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## Revenue Change
ttm_rev_chg = (ttm_rev - np.roll(ttm_rev, 1)) / np.roll(ttm_rev, 1)
ttm_rev_chg[0] = 0.
ann_rev_chg = (ann_rev - np.roll(ann_rev, 1)) / np.roll(ann_rev, 1)
ann_rev_chg[0] = 0.
#
fn = 'Revenue Change (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, ttm_rev_chg*100)
ax.set_xlabel('Date')
ax.set_ylabel('%')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Revenue Change (Ann)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_year, ann_rev_chg*100)
ax.set_xlabel('Date')
ax.set_ylabel('%')
ax.set_title(fn)
ax.tick_params(axis='x')
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## Total Assets and Total Liabilities
ass, _, _, _, _ = fa.get_fin_state_data('Total Assets', 'last', ttm=False)
lia, _, _, _, _ = fa.get_fin_state_data('Total Liabilities', 'last', ttm=False)
#
fn = 'Total Assets, Total Liabilities'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
_X = np.arange(len(rev_date))
ax.bar(_X-0.2, ass, 0.4, label='Total Assets')
ax.bar(_X+0.2, lia, 0.4, label='Total Liabilities')
ax.set_xticks(_X)
ax.set_xticklabels(rev_date)
ax.set_xlabel('Date')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## Current Ratio
curass, _, _, _, _ = fa.get_fin_state_data('Total Current Assets', 'last', ttm=False)
curlia, _, _, _, _ = fa.get_fin_state_data('Total Current Liabilities', 'last', ttm=False)
#
cur_ratio = curass / curlia
#
fn = 'Current Ratio'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, cur_ratio, label='CFO')
ax.set_xlabel('Date')
ax.set_ylabel('Ratio')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## CFO, CFI, CFF
cfo, ttm_cfo, _, ann_cfo, _ = fa.get_fin_state_data('Cash Flow From Operating Activities', 'sum', ttm=True)
cfi, ttm_cfi, _, ann_cfi, _ = fa.get_fin_state_data('Cash Flow From Investing Activities', 'sum', ttm=True)
cff, ttm_cff, _, ann_cff, _ = fa.get_fin_state_data('Cash Flow From Financial Activities', 'sum', ttm=True)
#
fn = 'CFO, -CFI, -CFF (Q)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.plot(rev_date, cfo, label='CFO')
ax.plot(rev_date, -cfi, label='-CFI')
ax.plot(rev_date, -cff, label='-CFF')
ax.set_xlabel('Date')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'CFO, -CFI, -CFF (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.plot(rev_date, ttm_cfo, label='CFO')
ax.plot(rev_date, -ttm_cfi, label='-CFI')
ax.plot(rev_date, -ttm_cff, label='-CFF')
ax.set_xlabel('Date')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'CFO, -CFI, -CFF (Ann)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
_X = np.arange(len(rev_year))
ax.bar(_X-0.3, ann_cfo, 0.3, label='CFO')
ax.bar(_X, -ann_cfi, 0.3, label='-CFI')
ax.bar(_X+0.3, -ann_cff, 0.3, label='-CFF')
ax.set_xticks(_X)
ax.set_xticklabels(rev_year)
ax.set_xlabel('Year')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x')
ax.grid()
ax.legend()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## Free Cash Flow
cppe, ttm_cppe, _, ann_cppe, _ = fa.get_fin_state_data('Net Change In Property, Plant, And Equipment', 'sum', ttm=True)
#
fcf = cfo + cppe
ttm_fcf = ttm_cfo + ttm_cppe
ann_fcf = ann_cfo + ann_cppe
#
fn = 'Free Cash Flow (Q)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, fcf)
ax.set_xlabel('Date')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Free Cash Flow (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, ttm_fcf)
ax.set_xlabel('Date')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Free Cash Flow (Ann)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_year, ann_fcf)
ax.set_xlabel('Year')
ax.set_ylabel('Million USD')
ax.set_title(fn)
ax.tick_params(axis='x')
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## LT Debt / Asset
ltdebt, _, _, ann_debt, _ = fa.get_fin_state_data('Long Term Debt', 'last', ttm=True)
#
ltdebt_ass = ltdebt / ass
#
fn = 'LT Debt - Asset Ratio'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, ltdebt_ass)
ax.set_xlabel('Date')
ax.set_ylabel('Ratio')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## LT Debt / Equity
equ, _, _, ann_equ, _ = fa.get_fin_state_data('Share Holder Equity', 'last', ttm=True)
#
ltdebt_equ = ltdebt / equ
#
fn = 'LT Debt Equity Ratio'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, ltdebt_equ)
ax.set_xlabel('Date')
ax.set_ylabel('Ratio')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## CFO / Net Profit
cfo_net = cfo / net
ttm_cfo_net = ttm_cfo / ttm_net
ann_cfo_net = ann_cfo / ann_net
#
fn = 'CFO - Net Profit Ratio (Q)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, cfo_net)
ax.set_xlabel('Date')
ax.set_ylabel('Ratio')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'CFO - Net Profit Ratio (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, ttm_cfo_net)
ax.set_xlabel('Date')
ax.set_ylabel('Ratio')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'CFO - Net Profit Ratio (Ann)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_year, ann_cfo_net)
ax.set_xlabel('Year')
ax.set_ylabel('Ratio')
ax.set_title(fn)
ax.tick_params(axis='x')
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## ROE
ttm_roe = ttm_net / equ
ann_roe = ann_net / ann_equ
#
fn = 'Return on Equity (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_date, ttm_roe * 100)
ax.set_xlabel('Date')
ax.set_ylabel('%')
ax.set_title(fn)
ax.tick_params(axis='x', rotation=90)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Return on Equity (Ann)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.bar(rev_year, ann_roe * 100)
ax.set_xlabel('Year')
ax.set_ylabel('%')
ax.set_title(fn)
ax.tick_params(axis='x')
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## EPS
eps, ttm_eps, _, ann_eps, _ = fa.get_fin_state_data('EPS - Earnings Per Share', 'sum', ttm=True)
#
fn = 'Earning per Share (Q)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax1 = plt.axes()
ax1.bar(rev_date, eps)
ax1.set_xlabel('Date')
ax1.set_ylabel('USD')
ax1.set_title(fn)
ax1.tick_params(axis='x', rotation=90)
ax1.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Earning per Share (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax1 = plt.axes()
ax1.bar(rev_date, ttm_eps)
ax1.set_xlabel('Date')
ax1.set_ylabel('USD')
ax1.set_title(fn)
ax1.tick_params(axis='x', rotation=90)
ax1.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')
#
fn = 'Earning per Share (Ann)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax2 = plt.axes()
ax2.bar(rev_year, ann_eps)
ax2.set_xlabel('Year')
ax2.set_ylabel('USD')
ax2.set_title(fn)
ax2.tick_params(axis='x')
ax2.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## Price related

price,pe, pfcf, peg, ticker_date = fa.get_valuation()

print('price: ' + str(price[-1]))
print('P/E: ' + str(pe[-1]))
print('P/FCF: ' + str(pfcf[-1]))
print('PEG: ' + str(peg[-1]))

fn = 'Price'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax1 = plt.axes()
ax1.plot(ticker_date, price)
ax1.set_xlabel('Date')
ax1.set_ylabel('USD')
ax1.set_title(fn)
ax1.grid()
ax2 = ax1.twinx()
ax2.plot(ticker_date, ((price / price[0]) - 1) * 100)
ax2.set_ylabel('%')
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

fn = 'PE Ratio (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.plot(ticker_date, pe)
ax.set_xlabel('Date')
ax.set_ylabel('USD')
ax.set_title(fn)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

fn = 'P - FCF Ratio (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.plot(ticker_date, pfcf)
ax.set_xlabel('Date')
ax.set_ylabel('USD')
ax.set_title(fn)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

fn = 'PEG (TTM)'
fig = plt.figure(dpi=150, figsize=(15, 5))
ax = plt.axes()
ax.plot(ticker_date, peg)
ax.set_xlabel('Date')
ax.set_ylabel('USD')
ax.set_title(fn)
ax.grid()
plt.savefig(out_dir + fn + '.png', bbox_inches='tight')
plt.close()
out_md = md_image(out_md, fn + '.png')

## write string to file
f = open(out_md_fn, 'w')
f.write(out_md)
f.close()