'''
20 years discounted cash flow.
It is developed by watching DCF computation videos by Adam Khoo.
Here the CF used is OCF, FCF, net income.
'''

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

## Retrieve company name from user input
parser = argparse.ArgumentParser()
parser.add_argument('company_name', type=str, help='Company name as in dcf_input_list.yaml')
args = parser.parse_args()

## Retrieve required input data from dcf_input_list.yaml
with open('dcf_input_list.yaml', 'r') as f:
    data = yaml.safe_load(f)
if 'fcf' in data[args.company_name]:
    fcf = data[args.company_name]['fcf']
if 'ocf' in data[args.company_name]:
    ocf = data[args.company_name]['ocf']
if 'net' in data[args.company_name]:
    net = data[args.company_name]['net']
date = data[args.company_name]['date']
debt = data[args.company_name]['debt']
cash = data[args.company_name]['cash']
growth_y1_y5 = 1 + data[args.company_name]['growth_y1_y5'] / 100.
growth_y6_y10 = 1 + data[args.company_name]['growth_y6_y10'] / 100.
growth_y11_y20 = 1 + data[args.company_name]['growth_y11_y20'] / 100.
shr_out = data[args.company_name]['shr_out']
dr = data[args.company_name]['dr']

## DCF function
def dcf_func(cf):
    div = 1. + dr/100.
    
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

    return int_val_per_share, cf_pred, pv_cf_pred

print('Company: ' + args.company_name + ' (' + date + ')')
if 'fcf' in data[args.company_name]:
    int_val_per_share, cf_pred, pv_cf_pred = dcf_func(fcf)
    print('Intrinsic value per share (FCF): ' + str(int_val_per_share))
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(np.arange(1, 21), cf_pred, label='projected CF')
    ax.plot(np.arange(1, 21), pv_cf_pred, label='discounted CF')
    ax.set_title('Based on FCF')
    ax.set_xlabel('Year')
    ax.set_ylabel('Million USD')
    ax.grid()
    ax.legend()
if 'ocf' in data[args.company_name]:
    int_val_per_share, cf_pred, pv_cf_pred = dcf_func(ocf)
    print('Intrinsic value per share (OCF): ' + str(int_val_per_share))
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(np.arange(1, 21), cf_pred, label='projected CF')
    ax.plot(np.arange(1, 21), pv_cf_pred, label='discounted CF')
    ax.set_title('Based on OCF')
    ax.set_xlabel('Year')
    ax.set_ylabel('Million USD')
    ax.grid()
    ax.legend()
if 'net' in data[args.company_name]:
    int_val_per_share, cf_pred, pv_cf_pred = dcf_func(net)
    print('Intrinsic value per share (Net): ' + str(int_val_per_share))
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(np.arange(1, 21), cf_pred, label='projected CF')
    ax.plot(np.arange(1, 21), pv_cf_pred, label='discounted CF')
    ax.set_title('Based on Net')
    ax.set_xlabel('Year')
    ax.set_ylabel('Million USD')
    ax.grid()
    ax.legend()

plt.show()