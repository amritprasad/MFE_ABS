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
# Create normalized ratio of unpaid balance
static_df['upb_ratio'] = static_df['Act_endg_upb'].divide(
        static_df['orig_upb'])
# Convert the percentage covariates to decimals
per_cols = ['orig_dti', 'orig_ltv', 'cpn_gap']
static_df[per_cols] /= 100
covar_cols = ['cpn_gap', 'summer']
covars = static_df[covar_cols].values
np.random.seed(42)
param = np.random.uniform(size=len(covar_cols) + 2)
tb = static_df['period_begin'].values/365
te = static_df['period_end'].values/365
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
hessian_inv = res.hess_inv.todense()
std_err = toolz.pipe(hessian_inv, np.diag, np.sqrt)
prop_std_err = (100*std_err/sol)

print('Initial LLK: {:.2f}'.format(-fnc.log_log_like(param, tb, te,
                                                     event, covars)))
print('Final LLK: {:.2f}'.format(-fnc.log_log_like(sol, tb, te,
                                                   event, covars)))
print('\nParameters:')
print('gamma =', gamma, '\np =', p, '\nCoupon Gap Coef =', beta[0],
      '\nSummer Indicator =', beta[1])
print('\nRespective Standard Errors:', ', '.join(std_err.round(2).astype(str)))
print('\nProportional Standard Errors:', ', '.join(prop_std_err.round(
        1).astype(str)))
# %%
# Plot the baseline hazard rate
t = np.linspace(0, 1, 100)
lambda_0 = gamma*p*((gamma*t)**(p-1))/(1 + (gamma*t)**p)
folder = 'Plots'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.scatter(t, lambda_0)
plt.grid(True)
plt.xlabel('Time (years)')
plt.ylabel('Baseline Hazard')
plt.title(r'$\lambda_0(t)$')
plt.savefig(os.path.join(folder, 'lambda_0.png'))
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
# The last row has a date missing. Fill it
zero_df.loc[zero_df.index[-1], 'DATE'] = zero_df.loc[
        zero_df.index[-2], 'DATE'] + pd.DateOffset(months=3)
# Calculate discount rates
discount_df = lib.discount_fac(zero_df)
# Calculate 10 yr rates
tenyr_df = fnc.calc_tenor_rate(discount_df, tenor=120)
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
param = np.array([14.5, 2.1, 22.5, -2.4])
tb = dynamic_df['period_begin'].values/365
te = dynamic_df['period_end'].values/365
event = dynamic_df['prepay'].values

eps = np.finfo(float).eps
bounds = ((eps, None), (eps, None), (None, None), (None, None))

# Run optimizer
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
hessian_inv_dyn = res_dyn.hess_inv.todense()
std_err_dyn = toolz.pipe(hessian_inv_dyn, np.diag, np.sqrt)
prop_std_err_dyn = (100*std_err_dyn/sol_dyn)

print('Initial LLK: {:.2f}'.format(-fnc.log_log_like(param, tb, te,
                                                     event, covars)))
print('Final LLK: {:.2f}'.format(-fnc.log_log_like(sol_dyn, tb, te,
                                                   event, covars)))
print('\nParameters:')
print('gamma =', gamma_dyn, '\np =', p_dyn, '\nCoupon Gap Coef =', beta_dyn[0],
      '\nSummer Indicator =', beta_dyn[1])
print('\nRespective Standard Errors:', ', '.join(
        std_err_dyn.round(2).astype(str)))
print('\nProportional Standard Errors:', ', '.join(prop_std_err_dyn.round(
        1).astype(str)))
# %%
# Plot the baseline hazard rate
t = np.linspace(0, 1, 100)
expo = np.append(np.nan, (gamma_dyn*t[1:])**(p_dyn-1))
lambda_0_dyn = gamma_dyn*p_dyn*expo/(1+(gamma_dyn*t)**p_dyn)
folder = 'Plots'
if not os.path.exists(folder):
    os.makedirs(folder)

plt.scatter(t, lambda_0_dyn)
plt.grid(True)
plt.xlabel('Time (years)')
plt.ylabel('Baseline Hazard')
plt.title(r'$\lambda_0(t)$')
plt.savefig(os.path.join(folder, 'lambda_0_dyn.png'))
# %%
