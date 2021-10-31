# fundamental_analysis

This repo contains:
1. Fundamental analysis script. <br/> This script computes and displays financial information of a company over time from its financial statements that are extracted from Macrotrends. It also computes some share price ratios. The price data is taken from yahoo finance.  
2. DCF (Discounted Cash Flow) calculator. <br/> 

<img src="output_example.gif" alt="output_example" width="1024"/>

## Tested Environment
- Python 3.8 and other packages in anaconda 2020.11
- yfinance 0.1.63

## How to use fundamental anaysis code
1. Open "fa_run.py" and modify all required information under "REQUIRED USER INPUTS".
2. Run "fa_run.py".
3. The result will be inside a folder named "output/..." where "..." is "output_prefix" in step 1 and followed by current date-time.
4. Open "0_fundamental_analysis.md" with browser (tested on firefox).

## How to use DCF calculator
1. Open "dcf.py" and modify all required information under "REQUIRED USER INPUTS".
2. Run "dcf.py".