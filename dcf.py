'''
20 years discounted cash flow.
It is developed by watching DCF computation videos by Adam Khoo.
In his videos, the CF used is the operating CF, but 
the free CF can be used instead to be more conservative.
'''

##########################################
########## REQUIRED USER INPUTS ##########
##########################################
'''
discount rate = risk free rate + beta * market risk premium
risk free rate --> usually from 10 year treasury rate

US stocks (from Adam Khoo's video)
risk free rate = 0.64%
average market risk premium = 5%
    Beta    Discount Rate
    <0.9    4.6%
    1       5.6%
    1.1     6.1%
    1.2     6.6%
    1.3     7.1%
    1.4     7.6%
    1.5     8.1%
    >1.6    8.6%

China/HK stocks (from Adam Khoo's video)
risk free rate = 0.6%
average market risk premium = 6.6%
    Beta    Discount Rate
    <0.9    5.9%
    1       7%
    1.1     7.9%
    1.2     9%
    1.3     9.2%
    1.4     10%
    1.5     10.5%
    >1.6    11%
''' 

## Alphabet 20210905 --> undervalued (3750.04)
cf = 58536  # in millions (FCF=58536, OCF=80859)
debt = 27984  # short + long term debt, in millions
cash = 135863  # cash and short term investment, in millions
growth_y1_y5 = 24.41  # in percent (from yfinance)
growth_y6_y10 = 12.2  # in percent
growth_y11_y20 = 1.72  # in percent (ave US GDP growth in trailing 20y = 1.72)
shr_out = 667.637  # share outstanding, in millions
dr = 7.  # discount rate, in percent (beta=1, risk free rate= 1.33%, market risk premium = 5.5%)

##########################################
############ PROCESSING CODE #############
##########################################

import numpy as np
import matplotlib.pyplot as plt

div = 1. + dr/100.
growth_y1_y5 = 1 + growth_y1_y5 / 100.
growth_y6_y10 = 1 + growth_y6_y10 / 100.
growth_y11_y20 = 1 + growth_y11_y20 / 100.

cf_pred = np.zeros(20)
pv_cf_pred = np.zeros(20)
for y in range(1, 5+1):
    idx = y - 1
    cf_pred[idx] = (cf * growth_y1_y5**y)
    pv_cf_pred[idx] = cf_pred[idx] / div**y
for y in range(6, 10+1):
    idx = y - 1
    cf_pred[idx] = (cf_pred[4] * growth_y6_y10**(y - 5))
    pv_cf_pred[idx] = cf_pred[idx] / div**y
for y in range(11, 20+1):
    idx = y - 1
    cf_pred[idx] = (cf_pred[9] * growth_y11_y20**(y - 10))
    pv_cf_pred[idx] = cf_pred[idx] / div**y

pv_20y = np.sum(pv_cf_pred)
int_val_per_share = (pv_20y - debt + cash) / shr_out
print('Intrinsic value per share: ' + str(int_val_per_share))

fig = plt.figure()
ax = plt.axes()
ax.plot(np.arange(1, 21), cf_pred, label='projected CF')
ax.plot(np.arange(1, 21), pv_cf_pred, label='discounted CF')
ax.set_xlabel('Year')
ax.set_ylabel('Million USD')
ax.set_title('PV of CF')
ax.grid()
ax.legend()

plt.show()
