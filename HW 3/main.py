"""
MFE 230M
Assignment 2
"""

# Imports
import pandas as pd
import numpy as np
import os
import sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import library
import abslibrary as lib  # HW 1 library
import abslibrary_2 as lib_2  # HW 2 library
import abslibrary_3 as lib_3  # HW 3 library

# Set options
pd.set_option('display.max_columns', 20)
# %%
# Verify that python version being used is 3.7
py_ver = sys.version.split(' ')
if py_ver[0][:3] != '3.7':
    raise OSError('Please install Python Version 3.7')
# %%
# Problem 1: Fit Hull-White Model
# Load LIBOR Data
data_folder = 'Data'
filename = 'BBG_Rates_20090630.xlsx'
filepath = os.path.join('.', data_folder, filename)
libor_df = pd.read_excel(filepath, sheet_name='Worksheet', header=0,
                         parse_dates=[0])

zero_df = libor_df[['Date', 'Zero Rate']].copy()
zero_df.columns = ['DATE', 'ZERO']
zero_df['ZERO'] /= 100.
# Drop rows with NA Zero rates
zero_df.dropna(subset=['ZERO'], inplace=True)

# Calculate discount rates
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
plt.savefig('./Plots/HW_3a_Discount.png')
plt.show()
# Calculate quarterly-compounded forward rates between each maturity date
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
plt.savefig('./Plots/HW_3b_Fwd_Rates.png')
plt.show()

# Load Flat Vol data
filename = 'Black Implied Vol.xlsx'
filepath = os.path.join('.', data_folder, filename)
blackvol_df = pd.read_excel(filepath, sheet_name='Quotes', header=0,
                            usecols=range(3))
blackvol_df.columns = ['EXPIRY', 'TYPE', 'FLAT_VOL']
blackvol_df['FLAT_VOL'] /= 100
blackvol_df['EXPIRY'].ffill(inplace=True)
blackvol_df['EXPIRY'] = blackvol_df['EXPIRY'].str.replace('Yr', '').astype(int)
atm_strike_df = blackvol_df[blackvol_df['TYPE'] == 'Strike'].drop(
        columns='TYPE').reset_index(drop=True)
atm_strike_df.rename(columns={'FLAT_VOL': 'ATM_STRIKE'}, inplace=True)
blackvol_df = blackvol_df[blackvol_df['TYPE'] == 'Vol'].drop(
        columns='TYPE').reset_index(drop=True)
# Calculate cap ATM swap rates
capswap_df = lib.capswap_rates(discount_df)
# Get maturity dates of caps for whom the Black Flat Vol is provided
settle_date = discount_df.index[0]
capdates = [settle_date + pd.DateOffset(years=x)
            for x in blackvol_df['EXPIRY']]
annual_idx = np.vectorize(capswap_df.index.get_loc)(capdates,
                                                    method='nearest')
# Get strikes for the annual caps
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
plt.savefig('./Plots/HW_3c_ATM_Cap_Strikes.png')
plt.show()

# Calculate caplet expiries by subtracting 2 good business days from the
# discount_df dates
caplet_expiry = lib_3.good_bday_offset(discount_df.index[1:], gbdays=-2)
swap_pay_dates = capswap_df.index
# Get Black prices of the caps
black_prices = np.zeros(blackvol_df.shape[0])
for i in range(blackvol_df.shape[0]):
    black_prices[i] = lib.black_price_caps(
            settle_date, blackvol_df['FLAT_VOL'].iloc[i],
            capatmstrike_df['CAPSWAP'].iloc[i],
            caplet_expiry[:annual_idx[i]],
            swap_pay_dates[:annual_idx[i]+1], discount_df, notional=1e2)

# Calibrate Hull-White model using the Black implied cap prices
guess = [0.1, 0.01]
eps = np.finfo(float).eps
bounds = ((eps, None), (eps, None))
res = minimize(lib.loss_hw_black, guess, args=(
        black_prices, annual_idx, swap_pay_dates, settle_date,
        capatmstrike_df, discount_df, 1e2), bounds=bounds)
if res.success:
    kappa, sigma = res.x
    print('kappa = %.2g\tsigma = %.2g' % (kappa, sigma))
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
plt.savefig('./Plots/HW_3d_Price_Comparison.png')
plt.show()

# Estimate θ(t)
ms_settle_date = pd.DatetimeIndex(start=settle_date, end=settle_date,
                                  periods=1).snap('MS')[0]
theta_df = lib.hw_theta(kappa, sigma, discount_df, ms_settle_date)
theta_df.to_csv('theta.csv')
t = np.vectorize(lib.t_dattime)(ms_settle_date, theta_df.index, 'ACTby365')
# Plot θ(t)
plt.figure(figsize=(10, 8))
plt.plot(t, theta_df.values.ravel(), marker='o', color='black')
plt.grid(True)
plt.title('θ(t) vs t')
plt.xlabel('Time (years)')
plt.ylabel(r'$\theta (t)$')
plt.savefig('./Plots/HW_3e_Theta.png')
plt.show()

# Calibrate Hull-White flat vols
hwvol_df = blackvol_df.copy()
for i in range(hw_prices.size):
    guess = hwvol_df['FLAT_VOL'].iloc[i]
    eps = np.finfo(float).eps
    bounds = ((eps, None),)
    res_flat = minimize(lib.loss_black_hw, guess, args=(
            hw_prices[i], caplet_expiry[:annual_idx[i]],
            swap_pay_dates[:annual_idx[i]+1], settle_date,
            capatmstrike_df['CAPSWAP'].iloc[i], discount_df, 1e2),
                   bounds=bounds)
    hwvol_df.loc[hwvol_df.index[i], 'FLAT_VOL'] = res_flat.x
    if not res_flat.x:
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
plt.savefig('./Plots/HW_3f_Flat_Vol_Comparison.png')
plt.show()
# %%
# Problem 2: Fitting Hazard Curves
# Specify whether prepayments data needs to filtered out for defaults and
# vice-versa
filt = False
# The fit_hazard function was run using basin-hopping and the parameters
# returned have been used as the guess for the standard minimize function.
# Basin-Hopping is a two-phase method that combines a global stepping algorithm
# with local minimization at each step. It gives values close to the global
# optima. Read more at:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping

# Fit ARM
data_folder = 'Data'
filename = 'ARM_perf.csv'
filepath = os.path.join('.', data_folder, filename)
data_df = pd.read_csv(filepath)
# Prepay
res_arm_p, sol_arm_p, hessian_inv_arm_p = lib_3.fit_hazard(
        data_df, prepay=True, filt=filt,
        guess=np.array([0.02591358, 1.55772621, -0.22301882, -0.01288632]))
# Default
res_arm_d, sol_arm_d, hessian_inv_arm_d = lib_3.fit_hazard(
        data_df, prepay=False, filt=filt,
        guess=np.array([0.01938579, 1.80575821, 0.66498693]))

# Fit FRM
filename = 'FRM_perf.csv'
filepath = os.path.join('.', data_folder, filename)
data_df = pd.read_csv(filepath)
# Prepay
res_frm_p, sol_frm_p, hessian_inv_frm_p = lib_3.fit_hazard(
        data_df, prepay=True, filt=filt,
        guess=np.array([0.01055477, 1.14627426, 0.05465934, -0.20479109]))
# Default
res_frm_d, sol_frm_d, hessian_inv_frm_d = lib_3.fit_hazard(
        data_df, prepay=False, filt=filt,
        guess=np.array([0.0060712, 1.4897565, 1.45588899]))
# %%
# Problem 3: Model cash flows

FRM_bal =  52416155
ARM_bal = 226122657
FRM_term = 354
FRM_age = 6
ARM_term = 354
ARM_age = 6
FRM_mwac = .07419/12
ARM_sprd = .055/12

#A1
#A2
#A3
#M1
#M2
#M3
#M4
#M5
#M6
#M7
#M8
#M9
#M10

orig_bal = 793431000
curr_bal = FRM_bal+ARM_bal

# A2-M8
curr_Tranche_bal=[
107769,
24954,
38481,
30150,
18646,
16265,
15075,
13488,
13092,
619
]

spreads=[
0.18,
0.28,
0.36,
0.38,
0.39,
0.51,
0.55,
0.62,
1.15,
1.4
]

# Fit hazards

# pool cash flow
    # Take interest rate path
    # Take home price paths
    # Compute hazards
    # Compute principal/interest/prepay/default CF
    # FRM
    # ARM

# Tranche cash flows
    # Per period, allocate principal, interest, prepay, default
    # Prepay starts at the top
    # Defaults starts with 1) excess spread, 2) OC, 3) bottom layer
    #   Extra principal distribution
    # Assume existence of a residual piece, which gets excess spread payments

