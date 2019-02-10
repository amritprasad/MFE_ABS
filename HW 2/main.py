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
import numdifftools as nd

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
# Problem 1
# Load Data
data_folder = 'Data'
filename = 'static.csv'
filepath = os.path.join('.', data_folder, filename)
static_df = pd.read_csv(filepath)

# Estimate static parameters
static_df['upb_ratio'] = static_df['Act_endg_upb'].divide(
        static_df['orig_upb'])
# Convert the percentage covariates to decimals
per_cols = ['orig_dti', 'orig_ltv', 'cpn_gap']
static_df[per_cols] /= 100
# covar_cols = ['orig_dti', 'orig_ltv', 'cpn_gap', 'summer', 'upb_ratio']
covar_cols = ['cpn_gap', 'summer']
covars = static_df[covar_cols].values
np.random.seed(42)
param = np.random.uniform(size=len(covar_cols) + 2)
# param = np.arange(0.01, 0.05, 0.01)
tb = static_df['period_begin'].values/365
te = static_df['period_end'].values/365
event = static_df['prepay'].values

eps = np.finfo(float).eps
# bounds = ([eps, None], [eps, None], [None, None], [None, None], [None, None],
#           [None, None], [None, None])
bounds = ((eps, None), (eps, None), (None, None), (None, None))
# Run optimizer
res = minimize(fun=fnc.log_log_like, x0=param, args=(tb, te, event, covars),
               jac=fnc.log_log_grad, bounds=bounds, method='L-BFGS-B')
print('Initial LLK: {:.2f}'.format(-fnc.log_log_like(param, tb, te,
                                                     event, covars)))
sol = res.x
std_err = np.sqrt(np.diag(res.hess_inv))
print('Final LLK: {:.2f}'.format(-fnc.log_log_like(sol, tb, te,
                                                   event, covars)))
print('Parameters:')
print('gamma =', sol[0], '\np =', sol[1], '\nCoupon Gap Coef =', sol[2],
      '\nSummer Indicator =', sol[3])
