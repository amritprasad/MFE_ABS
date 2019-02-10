"""
MFE 230M
Assignment 1
"""

# Imports
import pandas as pd
import numpy as np
import os
import sys
from scipy.optimize import fmin_tnc, minimize, Bounds
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
# Load LIBOR Data
data_folder = 'Data'
filename = 'static.csv'
filepath = os.path.join('.', data_folder, filename)
static_df = pd.read_csv(filepath)
static_df['upb_ratio'] = static_df['Act_endg_upb'].divide(
        static_df['orig_upb'])
# Convert the percentage covariates to decimals
per_cols = ['orig_dti', 'orig_ltv', 'cpn_gap']
static_df[per_cols] /= 100
# covar_cols = ['orig_dti', 'orig_ltv', 'cpn_gap', 'summer', 'upb_ratio']
covar_cols = ['cpn_gap', 'summer']
covars = static_df[covar_cols].values
param = np.random.uniform(size=len(covar_cols) + 2)
# param = np.arange(0.01, 0.05, 0.01)
tb = static_df['period_begin'].values.reshape(-1, 1)/365
te = static_df['period_end'].values.reshape(-1, 1)/365
event = static_df['prepay'].values.reshape(-1, 1)

tb = static_df['period_begin'].values/365
te = static_df['period_end'].values/365
event = static_df['prepay'].values

eps = np.finfo(float).eps
# bounds = ([eps, None], [eps, None], [None, None], [None, None], [None, None],
#           [None, None], [None, None])
bounds = ([eps, None], [eps, None], [None, None], [None, None])
bounds = ((eps, None), (eps, None), (None, None), (None, None))
# Run optimizer
sol, nfeval, rc = fmin_tnc(func=fnc.log_log_like, x0=param,
                           fprime=fnc.log_log_grad, args=(
                                   tb, te, event, covars),
                           disp=5, bounds=bounds, ftol=eps)
res = minimize(fun=fnc.log_log_like, x0=param, args=(tb, te, event, covars),
               jac=fnc.log_log_grad, bounds=bounds)
# std_err = np.apply_along_axis(lambda x: np.std(x, ddof=1)/np.sqrt(x.size),
#                               axis=0, arr=np.stack(fnc.phist))
H = nd.Hessian(fnc.log_log_like, step=1e-3)(sol, tb, te, event, covars)
std_err = np.sqrt(np.diag(np.linalg.inv(H))/len(covars))
print('Parameters:')
print('gamma =', sol[0], '\np =', sol[1], '\nCoupon Gap Coef =', sol[2],
      '\nSummer Indicator =', sol[3])
