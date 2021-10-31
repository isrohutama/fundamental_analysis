# fundamental_analysis

This repo contains:
1. Fundamental analysis script.<br/>This script computes and displays financial information of a company over time from its financial statements that are extracted from Macrotrends. It also computes some share price ratios. The price data is taken from yahoo finance.  
2. DCF (Discounted Cash Flow) calculator. 

<img src="output_example.gif" alt="output_example" width="1024"/>

## Tested Environment
- Python 3.8 and other packages in anaconda 2020.11
- yfinance 0.1.63

## How to use fundamental anaysis script
1. Open "fa_input_list.yaml".
2. Search the company name and update its information if it is needed.<br/> If the company name is not in the list, create a new entry and fill all required information (You can use other company entry available as a template).
2. Run "python fa_run.py <the_company_name>". It must be executed from the same directory as "fa_run.py". 
3. The result will be inside a folder named "output/..." where "..." is "output_prefix" in step 1 and followed by current date-time.
4. Open "0_fundamental_analysis.md" with browser (tested on firefox).

## How to use DCF calculator
1. Open "dcf_input_list.yaml".
2. Search the company name and update its information if it is needed.<br/> If the company name is not in the list, create a new entry and fill all required information (You can use other company entry available as a template).
3. Run "python dcf.py <the_company_name>". It must be executed from the same directory as "dcf.py". 