"""
MFE 230M
Assignment 1
"""

# Imports
import pandas as pd
import numpy as np
import os
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import library
import library as lib

# Set options
pd.set_option('display.max_columns', 10)
# %%
# Verify that python version being used is 3.7
py_ver = sys.version.split(' ')
if py_ver[0][:3] != '3.7':
    raise OSError('Please install Python Version 3.7')
# %%
# Problem 1
# Load LIBOR Data
data_folder = 'Data'
filename = '20040830_usd23_atm_caps_jan_2019_update2.xlsx'
filepath = os.path.join('.', data_folder, filename)
libor_df = pd.read_excel(filepath, sheet_name='usd23_libor_curve', skiprows=3,
                         header=0, parse_dates=[0, 1])

zero_df = libor_df[['Date', 'Semi-Compounded Zero Rate (%)']].copy()
zero_df.columns = ['DATE', 'ZERO']
zero_df['ZERO'] /= 100.
# Drop rows with NA Zero rates
zero_df.dropna(subset=['ZERO'], inplace=True)
# The last row has a date missing. Fill it
zero_df.loc[zero_df.index[-1], 'DATE'] = zero_df.loc[
        zero_df.index[-2], 'DATE'] + pd.DateOffset(months=3)

# a) Calculate discount rates
discount_df = lib.discount_fac(zero_df)
# b) Calculate quarterly-compounded forward rates between each maturity date
fwd_df = lib.fwd_rates(discount_df)
# %%
# Load Flat Vol data
blackvol_df = pd.read_excel(filepath, sheet_name='usd_atm_european_caps',
                            skiprows=2, header=0)
blackvol_df.columns = ['EXPIRY', 'FLAT_VOL']
blackvol_df['FLAT_VOL'] /= 100
blackvol_df['EXPIRY'] = blackvol_df['EXPIRY'].str.replace('Yr', '').astype(int)
# Calculate cap swap rates
capswap_df = lib.capswap_rates(discount_df)
# Get maturity dates of caps for whom the Black Flat Vol is provided
settle_date = discount_df.index[0]
capdates = [settle_date + pd.DateOffset(years=x)
            for x in blackvol_df['EXPIRY']]
annual_idx = np.vectorize(capswap_df.index.get_loc)(capdates,
                                                    method='nearest')
# c) Get strikes for the annual caps
capatmstrike_df = capswap_df.iloc[annual_idx]

# Get good business days
caplet_expiry = libor_df['Caplet Accrual Expiry Date'].dropna()[1:]
swap_pay_dates = capswap_df.index
# Get Black prices of the caps
black_prices = np.zeros(15)
for i in range(15):
    black_prices[i] = lib.black_price_caps(
            settle_date, blackvol_df['FLAT_VOL'].iloc[i],
            capatmstrike_df['CAPSWAP'].iloc[i],
            caplet_expiry[:annual_idx[i]],
            swap_pay_dates[:annual_idx[i]+1], discount_df, notional=1e2)

# d) Calibrate Hull-White model using the Black implied cap prices
guess = [0.1, 0.01]
eps = np.finfo(float).eps
bounds = ((eps, None), (eps, None))
res = minimize(lib.loss_hw_black, guess, args=(
        black_prices, annual_idx, swap_pay_dates, settle_date,
        capatmstrike_df, discount_df, 1e2), bounds=bounds)
if res.success:
    kappa, sigma = res.x
else:
    raise ValueError("Optimizer didn't converge")

# Get Hull-White cap prices
hw_prices = np.zeros(black_prices.size)
for i in range(hw_prices.size):
    hw_prices[i] = lib.hullwhite_price_caps(
            settle_date, sigma, kappa, capatmstrike_df['CAPSWAP'].iloc[i],
            swap_pay_dates[:annual_idx[i]+1], discount_df,
            notional=1e2)

# Plot the Black Cap prices
plt.figure(figsize=(10, 8))
plt.plot(pd.Series(black_prices, index=blackvol_df['EXPIRY']), marker='o',
         color='black')
plt.plot(pd.Series(hw_prices, index=blackvol_df['EXPIRY']), marker='+',
         color='red')
plt.grid(True)
plt.legend(['Black', 'Hull-White'])
plt.title('Comparison between Black and Hull-White Cap Prices')
plt.xlabel('Cap Expiry')
plt.ylabel(r'$\$$ Amount per $\$100$ Notional')
if not os.path.exists('./Plots'):
    os.makedirs('./Plots')
plt.savefig('./Plots/HW_1d_Price_Comparison.png')
plt.show()
