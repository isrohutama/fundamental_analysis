# fundamental_analysis

This repo contains:
1. Fundamental analysis script.<br/>This script computes and displays financial information of a company over time from its financial statements that are extracted from Macrotrends. It also computes some share price ratios. The price data is taken from yahoo finance.  
2. DCF (Discounted Cash Flow) calculator. 

<img src="output_example.gif" alt="output_example" width="1024"/>

## Environment Setup Guide
- Install miniconda
- Open terminal for Linux or "anaconda prompt" for Windows
- Create a new conda environment: `conda env create --name finenv`
- Enter the newly created environment: `conda activate finenv`
- Install pip: `conda install pip`
- Install other required packages using pip:
  - `pip install numpy`
  - `pip install matplotlib`
  - `pip install pandas`
  - `pip install bs4`
  - `pip install lxml`
  - `pip install PyYAML`
  - `pip install yfinance`

## How to use fundamental anaysis script
1. Open "fa_input_list.yaml".
2. Search the company name and update its information if it is needed.<br/> If the company name is not in the list, create a new entry and fill all required information (You can use other company entry available as a template).
2. Run "python fa_run.py <the_company_name>" using terminal with "finenv" environment activated. It must be executed from the same directory as "fa_run.py". 
3. The result will be inside a folder named "output/..." where "..." is "output_prefix" in step 1 and followed by current date-time.
4. Open "0_<the_company_name>.md" using chrome with "Markdown Viewer" extension ("Allow access to file URLs" setting must be enabled).

## How to use DCF calculator
1. Open "dcf_input_list.yaml".
2. Search the company name and update its information if it is needed.<br/> If the company name is not in the list, create a new entry and fill all required information (You can use other company entry available as a template).
3. Run "python dcf.py <the_company_name>". It must be executed from the same directory as "dcf.py". 