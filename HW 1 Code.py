"""
MFE 230M
Assignment 1
"""

# Imports
import pandas as pd
import numpy as np
import os

# Import library
import library as lib

# Set options
pd.set_option('display.max_columns', 10)
# %%
# Verify that python version used is 3.7+
py_ver = !python --version
if py_ver[0].split(' ')[-1][:3] != '3.7':
    raise OSError('Please install Python Version 3.7+')
# %%
# Problem 1
# Load LIBOR Data
data_folder = 'Data'
filename = '20040830_usd23_atm_caps_jan_2019_update2.xlsx'
filepath = os.path.join('.', data_folder, filename)
libor_df = pd.read_excel(filepath, sheet_name='usd23_libor_curve', skiprows=3,
                         header=0, parse_dates=[0, 1]).dropna()

zero_df = libor_df[['Date', 'Semi-Compounded Zero Rate (%)']].copy()
zero_df.columns = ['DATE', 'ZERO']
zero_df['ZERO'] /= 100.

# Calculate discount rates
disc_df = lib.discount_fac(zero_df)
