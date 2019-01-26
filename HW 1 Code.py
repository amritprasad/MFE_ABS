"""
MFE 230M
Assignment 1
"""

# Imports
import pandas as pd
import numpy as np
import os

# Set options
pd.set_option('display.max_columns', 10)
# %%


def t_30by360(d0, d1):
    """
    Function to calculate 30/360 daycount time

    Args:
        d0, d1 (pd.Timestamp)

    Returns:
        float
    """
    days_30by360 = 360*(d1.year - d0.year) + 30*(d1.month - d0.month) +\
        (d1.day - d0.day)
    return days_30by360/360


def discount_fac(_zero_df):
    """
    Function to calculate the discount rates given the zero rates. Columns
    should be DATE and ZERO

    Args:
        _zero_df (pd.DataFrame)

    Returns:
        pd.DataFrame containing discount factors
    """
    zero_df = _zero_df.copy()
    # Convert DATE to pandas Timestamp
    zero_df['DATE'] = zero_df['DATE'].astype(pd.Timestamp)
    T_i = np.vectorize(t_30by360)(zero_df['DATE'][:-1], zero_df['DATE'][1:])
    discount_df = pd.DataFrame(((1 + zero_df['ZERO'][1:]/2)**(-2*T_i)).values,
                               index=zero_df['DATE'][1:],
                               columns=['DISCOUNT'])
    discount_df.loc[zero_df['DATE'].iloc[0]] = 1.
    discount_df.sort_index(inplace=True)
    return discount_df


# %%
# Load Data
data_folder = 'Data'
filename = '20040830_usd23_atm_caps_jan_2019_update2.xlsx'
filepath = os.path.join('.', data_folder, filename)
libor_df = pd.read_excel(filepath, sheet_name='usd23_libor_curve', skiprows=3,
                         header=0, parse_dates=[0, 1]).dropna()
zero_df = libor_df[['Date', 'Semi-Compounded Zero Rate (%)']].copy()
zero_df.columns = ['DATE', 'ZERO']
zero_df['ZERO'] /= 100.
