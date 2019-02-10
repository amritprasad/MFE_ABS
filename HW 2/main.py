"""
MFE 230M
Assignment 1
"""

# Imports
import pandas as pd
import numpy as np
import os
import sys
from scipy.optimize import fmin_tnc
import matplotlib.pyplot as plt

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
tb = static_df['period_begin'].values/365
te = static_df['period_end'].values/365
event = static_df['prepay'].values
fnc.log_log_like(param, tb, te, event, covars)
fnc.log_log_grad(param, tb, te, event, covars)
fmin_tnc(func=fnc.log_log_grad, x0=param, args=(tb, te, event, covars), disp=5)
