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
import toolz

# Import library
import abslibrary as lib  # HW 1 library
import function as fnc  # HW 2 library

# Set options
pd.set_option('display.max_columns', 20)
# %%
# Verify that python version being used is 3.7
py_ver = sys.version.split(' ')
if py_ver[0][:3] != '3.7':
    raise OSError('Please install Python Version 3.7')
# %%
# Static Estimation
# a)
# Load Data
data_folder = 'Data'
filename = 'static.csv'
filepath = os.path.join('.', data_folder, filename)
static_df = pd.read_csv(filepath)
# Estimate static parameters
# Convert the percentage covariates to decimals
per_cols = ['orig_dti', 'orig_ltv', 'cpn_gap']
static_df[per_cols] /= 100
covar_cols = ['cpn_gap', 'summer']
covars = static_df[covar_cols].values
np.random.seed(42)
param = np.random.uniform(size=len(covar_cols) + 2)
# param = np.array([0.01, 0.02, 0.2, 0.04])
# Since the times are in months, convert them to yearly time to make them the
# same units as the coupon gap
tb = static_df['period_begin'].values
te = static_df['period_end'].values
event = static_df['prepay'].values

eps = np.finfo(float).eps
bounds = ((eps, None), (eps, None), (None, None), (None, None))
# Run optimizer
res = minimize(fun=fnc.log_log_like, x0=param, args=(tb, te, event, covars),
               jac=fnc.log_log_grad, bounds=bounds, method='L-BFGS-B')
if not res.success:
    raise ValueError('Optimizer did not converge')

sol = res.x
gamma, p, _, _ = sol
beta = sol[2:]
# Calculate standard errors as the square root of the diagonal elements of the
# Hessian inverse
N = len(covars)
hessian_inv = res.hess_inv.todense()
std_err = toolz.pipe(hessian_inv/N, np.diag, np.sqrt)
prop_std_err = (100*std_err/sol)

print('Initial LLK: {:.2f}'.format(-fnc.log_log_like(param, tb, te,
                                                     event, covars)))
print('Final LLK: {:.2f}'.format(-fnc.log_log_like(sol, tb, te,
                                                   event, covars)))
print('\nParameters:')
print('gamma =', gamma, '\np =', p, '\nCoupon Gap Coef =', beta[0],
      '\nSummer Indicator =', beta[1])
print('\nRespective Standard Errors:', ', '.join(std_err.round(3).astype(str)))
print('\nProportional Standard Errors:', ', '.join(prop_std_err.round(
        1).astype(str)))

# %%
# Plot the baseline hazard rate for 10 years
t = np.linspace(0, 120, 500)
lambda_0 = gamma*p*((gamma*t)**(p-1))/(1 + (gamma*t)**p)
folder = 'Plots'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.scatter(t, lambda_0)
plt.grid(True)
plt.xlabel('Time (months)')
plt.ylabel('Baseline Hazard')
plt.title(r'$\lambda_0(t)$')
plt.savefig(os.path.join(folder, 'lambda_0.png'))
plt.close()
# %%
# b), c)
# Get Hull-White parameters from HW 1
kappa, sigma = (0.1141813928341348, 0.01453600529450289)
# Get discount factors
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
# The last row has a date missing. Fill itlibor_df
zero_df.loc[zero_df.index[-1], 'DATE'] = zero_df.loc[
        zero_df.index[-2], 'DATE'] + pd.DateOffset(months=3)
# Calculate discount rates
discount_df = lib.discount_fac(zero_df)
# Calculate theta_df
theta_df = lib.hw_theta(kappa, sigma, discount_df, discount_df.index.min())


# %%
# Calculate Cash Flows for the two Mtge Poolscou

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

bond_list = ['CG','VE','CM','GZ','TC','CZ','CA','CY','R']
Tranche_bal_arr = np.array([
                         74800000, #CG
                         5200000,  #VE
                         14000000, #CM
                         22000000, #GZ
                         20000000, #TC
                         24000000, #CZ
                         32550000, #CA
                         13950000  #CY
                         ])# initial Balance

wac = (Pool1_wac+Pool2_wac)/2

m = 10000
tenor = 120
antithetic = True

settle_date = discount_df.index[0]
theta_df = lib.hw_theta(kappa, sigma, discount_df, settle_date)

r0 = np.log((zero_df.iloc[1, 1]/2+1)**(0.5))/0.25

price_df, smm_df = fnc.mc_bond(m, theta_df, kappa, sigma, gamma, p, beta, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                    Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate)
price_df = pd.DataFrame(price_df, columns=bond_list)
price_mean = price_df.mean()
price_std = price_df.std()/np.sqrt(len(price_df))


# %% Calculate Duration, Convexity, and OAS
duration, convexity = fnc.calc_duration_convexity(m, theta_df, kappa, sigma, gamma, p, beta, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                                                 Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate)

oas = fnc.calc_OAS(m, theta_df, kappa, sigma, gamma, p, beta, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                   Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate)

# %%
# Dynamic Estimation
# d)
# Load Data
data_folder = 'Data'
filename = 'dynamic.csv'
filepath = os.path.join('.', data_folder, filename)
dynamic_df = pd.read_csv(filepath)
# Convert the percentage covariates to decimals
per_cols = ['orig_dti', 'orig_ltv', 'cpn_gap']
dynamic_df[per_cols] /= 100
covar_cols = ['cpn_gap', 'summer']
covars = dynamic_df[covar_cols].values
# Provide guess = static estimation
param = np.array([0.013, 1.37, 0.72, 0.23])
tb = dynamic_df['period_begin'].values
te = dynamic_df['period_end'].values
event = dynamic_df['prepay'].values

eps = np.finfo(float).eps
bounds = ((eps, None), (eps, None), (None, None), (None, None))

# Run optimizer. WARNING!!! MIGHT TAKE UPTO 15 min depending upon config
res_dyn = minimize(fun=fnc.log_log_like, x0=param, args=(
        tb, te, event, covars), jac=fnc.log_log_grad, bounds=bounds,
                   method='L-BFGS-B', options={'disp': True})
if not res_dyn.success:
    raise ValueError('Optimizer did not converge')

sol_dyn = res_dyn.x
gamma_dyn, p_dyn, _, _ = sol_dyn
beta_dyn = sol_dyn[2:]
# Calculate standard errors as the square root of the diagonal elements of the
# Hessian inverse
N = len(covars)
hessian_inv_dyn = res_dyn.hess_inv.todense()
std_err_dyn = toolz.pipe(hessian_inv_dyn/N, np.diag, np.sqrt)
prop_std_err_dyn = (100*std_err_dyn/sol_dyn)

print('Initial LLK: {:.2f}'.format(-fnc.log_log_like(param, tb, te,
                                                     event, covars)))
print('Final LLK: {:.2f}'.format(-fnc.log_log_like(sol_dyn, tb, te,
                                                   event, covars)))
print('\nParameters:')
print('gamma =', gamma_dyn, '\np =', p_dyn, '\nCoupon Gap Coef =', beta_dyn[0],
      '\nSummer Indicator =', beta_dyn[1])
print('\nRespective Standard Errors:', ', '.join(
        std_err_dyn.round(3).astype(str)))
print('\nProportional Standard Errors:', ', '.join(prop_std_err_dyn.round(
        1).astype(str)))
# %%
# Plot the baseline hazard rate for 10 years
t = np.linspace(0, 120, 500)
expo = (gamma_dyn*t)**(p_dyn-1)
lambda_0_dyn = gamma_dyn*p_dyn*expo/(1+(gamma_dyn*t)**p_dyn)
folder = 'Plots'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.clf()
plt.scatter(t, lambda_0_dyn)
plt.grid(True)
plt.xlabel('Time (months)')
plt.ylabel('Baseline Hazard')
plt.title(r'$\lambda_0(t)$')
plt.savefig(os.path.join(folder, 'lambda_0_dyn.png'))
plt.close()
# %%
#e)
price_df_dyn, smm_dyn_df = fnc.mc_bond(m, theta_df, kappa, sigma, gamma_dyn, p_dyn, beta_dyn, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                           Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate)
price_df_dyn = pd.DataFrame(price_df_dyn, columns=bond_list)
price_mean_dyn = price_df_dyn.mean()
price_std_dyn = price_df_dyn.std()/np.sqrt(len(price_df_dyn))


# %% Calculate Duration, Convexity, and OAS
duration_dyn, convexity_dyn = fnc.calc_duration_convexity(m, theta_df, kappa, sigma, gamma_dyn, p_dyn, beta_dyn, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                                                          Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate)

oas_dyn = fnc.calc_OAS(zero_df, m, theta_df, kappa, sigma, gamma_dyn, p_dyn, beta_dyn, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                   Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate)
# %%
# Plot the hazard rates
folder = 'Plots'
filename = 'Hazard_Rate.png'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.figure(figsize=(10, 8))
plt.plot(smm_df.mean(axis=1), label='Mean Static')
plt.plot(smm_dyn_df.mean(axis=1), label='Mean Dynamic')
plt.legend()
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Hazard Rate')
plt.savefig(os.path.join(folder, filename))
plt.show()
plt.close()
# %%
