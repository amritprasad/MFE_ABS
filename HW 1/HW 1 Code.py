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
import abslibrary as lib

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
# Plot the discount rates
plt.figure(figsize=(10, 8))
plt.plot(discount_df, marker='o', color='black')
plt.grid(True)
plt.title('Discount Factors')
plt.xlabel('Dates')
plt.ylabel('Discount')
if not os.path.exists('./Plots'):
    os.makedirs('./Plots')
plt.savefig('./Plots/HW_1a_Discount.png')
plt.show()
# b) Calculate quarterly-compounded forward rates between each maturity date
fwd_df = lib.fwd_rates(discount_df)
# Plot the forward rates
plt.figure(figsize=(10, 8))
plt.plot(fwd_df, marker='o', color='black')
plt.grid(True)
plt.title('Forward Rates')
plt.xlabel('Dates')
plt.ylabel('Fwd Rates')
if not os.path.exists('./Plots'):
    os.makedirs('./Plots')
plt.savefig('./Plots/HW_1b_Fwd_Rates.png')
plt.show()
# %%
# Load Flat Vol data
blackvol_df = pd.read_excel(filepath, sheet_name='usd_atm_european_caps',
                            skiprows=2, header=0)
blackvol_df.columns = ['EXPIRY', 'FLAT_VOL']
blackvol_df['FLAT_VOL'] /= 100
blackvol_df['EXPIRY'] = blackvol_df['EXPIRY'].str.replace('Yr', '').astype(int)
# Calculate cap ATM swap rates
capswap_df = lib.capswap_rates(discount_df)
# Get maturity dates of caps for whom the Black Flat Vol is provided
settle_date = discount_df.index[0]
capdates = [settle_date + pd.DateOffset(years=x)
            for x in blackvol_df['EXPIRY']]
annual_idx = np.vectorize(capswap_df.index.get_loc)(capdates,
                                                    method='nearest')
# c) Get strikes for the annual caps
capatmstrike_df = capswap_df.iloc[annual_idx]
# Plot the ATM cap strikes
plt.figure(figsize=(10, 8))
plt.plot(capatmstrike_df, marker='o', color='black')
plt.grid(True)
plt.title('ATM Cap Strikes')
plt.xlabel('Dates')
plt.ylabel('Cap Strikes')
if not os.path.exists('./Plots'):
    os.makedirs('./Plots')
plt.savefig('./Plots/HW_1c_ATM_Cap_Strikes.png')
plt.show()

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
# %%
# e) Estimate θ(t)
theta_df = lib.hw_theta(kappa, sigma, discount_df, settle_date)
theta_df.to_csv('theta.csv')
t = np.vectorize(lib.t_dattime)(settle_date, theta_df.index, 'ACTby365')
# Plot θ(t)
plt.figure(figsize=(10, 8))
plt.plot(t, theta_df.values.ravel(), marker='o', color='black')
plt.grid(True)
plt.title('θ(t) vs t')
plt.xlabel('Time (years)')
plt.ylabel(r'$\theta (t)$')
plt.savefig('./Plots/HW_1e_Theta.png')
plt.show()
# %%
# f) Calibrate Hull-White flat vols
hwvol_df = blackvol_df.copy()
for i in range(hw_prices.size):
    guess = hwvol_df['FLAT_VOL'].iloc[i]
    eps = np.finfo(float).eps
    bounds = ((eps, None),)
    res = minimize(lib.loss_black_hw, guess, args=(
            hw_prices[i], caplet_expiry[:annual_idx[i]],
            swap_pay_dates[:annual_idx[i]+1], settle_date,
            capatmstrike_df['CAPSWAP'].iloc[i], discount_df, 1e2),
                   bounds=bounds)
    hwvol_df.loc[hwvol_df.index[i], 'FLAT_VOL'] = res.x
    if not res.x:
        print(i)
# Plot the vols
plt.figure(figsize=(10, 8))
plt.plot(blackvol_df['EXPIRY'], blackvol_df['FLAT_VOL'], marker='o',
         color='black')
plt.plot(hwvol_df['EXPIRY'], hwvol_df['FLAT_VOL'], marker='+',
         color='red')
plt.grid(True)
plt.legend(['Black', 'Hull-White'])
plt.title('Comparison between Black and Hull-White Cap Flat Vols')
plt.xlabel('Cap Expiry')
plt.ylabel('Flat Volatility')
plt.savefig('./Plots/HW_1f_Flat_Vol_Comparison.png')
plt.show()


# %%
# Problem 2
# %%
# Hard coded parameters

# Starting Balance
Pool1_bal = 77657656.75
Pool2_bal = 128842343.35
Pool1_term = 236
Pool2_term = 237

# Coupon rates
Pool1_wac = 0.05402
Pool2_wac = 0.05419
Pool1_mwac = Pool1_wac/12
Pool2_mwac = Pool2_wac/12
Pool1_age = 3
Pool2_age = 3
coupon_rate = .05/12

# PSA - constant (not dependent on interest rate)
PSA = 1.50

# %%
# Calculate Cash Flows for the two Mtge Pools
CPR_array = lib.cpr(PSA, .06, 241, 3)  # hard coded parameters
SMM_array = 1 - (1-CPR_array)**(1/12)
assert(False)
# pre-allocate arrays to hold pool cash flows
# 7 columns: PMT, Interest, Principal, Pre-pmt CPR, SMM, pp CF, Balance
cols = ['PMT', 'Interest', 'Principal', 'PP CF', 'Balance']
Pool1_data = pd.DataFrame(np.zeros((241, 5)), columns=cols)
Pool2_data = pd.DataFrame(np.zeros((241, 5)), columns=cols)

Pool1_data.loc[0, 'Balance'] = Pool1_bal
Pool2_data.loc[0, 'Balance'] = Pool2_bal
lib.pool_cf(Pool1_data, Pool1_mwac, Pool1_age, Pool1_term, SMM_array)
lib.pool_cf(Pool2_data, Pool2_mwac, Pool2_age, Pool2_term, SMM_array)

# %%
# Reproduce Principal CF Allocation (waterfall)
Total_principal = Pool1_data['Principal'] + Pool2_data[
        'Principal'] + Pool1_data['PP CF'] + Pool2_data['PP CF']
CA_CY_princ = 0.225181598 * Total_principal
Rest_princ = 0.7748184 * Total_principal

# Pre-Allocate for each Tranche, then do waterfall logic
Tranche_dict = {}
Tranche_list = ['CG', 'VE', 'CM', 'GZ', 'TC', 'CZ', 'CA', 'CY']
cols = ['Principal', 'Interest', 'Balance']

Tranche_bal_dict = {
                    'CG': 74800000,
                    'VE': 5200000,
                    'CM': 14000000,
                    'GZ': 22000000,
                    'TC': 20000000,
                    'CZ': 24000000,
                    'CA': 32550000,
                    'CY': 13950000
                   }

# small for loop, pre-allocate data for tranches
for i in Tranche_list:
    Tranche_dict[i] = pd.DataFrame(np.zeros((241, 3)), columns=cols)
    Tranche_dict[i].loc[0, 'Balance'] = Tranche_bal_dict[i]

# Temporary arrays for calculating GZ/CZ interest accrual. The "interest" here
# is not cash flow.
cols = ['interest', 'accrued']
GZ_interest = pd.DataFrame(np.zeros((241, 2)), columns=cols)
CZ_interest = pd.DataFrame(np.zeros((241, 2)), columns=cols)

# Calculate Cash flows
lib.tranche_CF_calc(Tranche_dict, CA_CY_princ, Rest_princ, GZ_interest,
                    CZ_interest, coupon_rate)

CF_df = pd.DataFrame(np.zeros((240, 8)), columns=list(Tranche_bal_dict.keys()))
for i in Tranche_bal_dict.keys():
    CF_df[i] = Tranche_dict[i]['Principal'] + Tranche_dict[i]['Interest']

# %% Calculate Standard Errors, Duration, Convexity, and OAS
cf_bond = CF_df.iloc[1:, :]

m = 10000

r0 = np.log((zero_df.iloc[1, 1]/2+1)**(0.5))/0.25
price_df_MC = lib.mc_bond(m, cf_bond, theta_df, kappa, sigma, r0,
                          antithetic=True)
price_mean_MC = price_df_MC.mean()
price_std_MC = price_df_MC.std()/np.sqrt(len(price_df_MC))

duration, convexity = lib.calc_duration_convexity(m, cf_bond, theta_df, kappa,
                                                  sigma, r0, antithetic=True)

oas_ser = lib.calc_OAS(cf_bond, zero_df)
