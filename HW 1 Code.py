"""
MFE 230M
Assignment 1
"""

# Imports
import pandas as pd
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


def discount_fac(zero_df):
    """
    Function to calculate the discount rates given the zero rates. Index should
    be DATE and one of the columns should be ZERO

    Args:
        zero_df (pd.DataFrame)

    Returns:
        pd.DataFrame containing discount factors
    """
    return


# %%
# Load Data
data_folder = 'Data'
filename = '20040830_usd23_atm_caps_jan_2019_update2.xlsx'
filepath = os.path.join('.', data_folder, filename)
libor_df = pd.read_excel(filepath, sheet_name='usd23_libor_curve', skiprows=3,
                         header=0, parse_dates=[0, 1]).dropna()
