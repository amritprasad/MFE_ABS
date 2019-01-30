"""
MFE 230M
Assignment 1
"""

# Imports
import pandas as pd
import numpy as np
import os
import sys

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
goodbday = capswap_df.index
caplet_mat = libor_df['Caplet Accrual Expiry Date'].dropna()
