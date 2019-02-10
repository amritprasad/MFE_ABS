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
import toolz as t

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
std_err = t.pipe(hessian_inv, np.diag, np.sqrt)
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
# b)
# Get Hull-White parameters from HW 1
kappa, sigma = (0.1141813928341348, 0.01453600529450289)
