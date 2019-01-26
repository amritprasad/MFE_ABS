"""
MFE 230M
Library of functions
"""

# Imports
import pandas as pd
import numpy as np


def t_30by360(d0, d1):
    """
    Function to calculate 30/360 daycount time in annual terms

    Args:
        d0, d1 (pd.Timestamp)

    Returns:
        float
    """
    days_30by360 = 360*(d1.year - d0.year) + 30*(d1.month - d0.month) +\
        (d1.day - d0.day)
    return days_30by360/360


def discount_fac(_zero_df, incremental=False):
    """
    Function to calculate the discount rates given the zero rates. Columns
    should be DATE and ZERO

    Args:
        _zero_df (pd.DataFrame)

        incremental (bool): specify if the siscount factors need to be
        calculated between index_t-1 and index_t or index_min and index_t

    Returns:
        pd.DataFrame containing discount factors. Index is DATE and DISCOUNT
        contains discount factors
    """
    # Create deep copy to avoid modifying the original df passed
    zero_df = _zero_df.copy()
    # Convert DATE to pandas Timestamp
    zero_df['DATE'] = zero_df['DATE'].astype(pd.Timestamp)
    # Calculate T_i
    T_i = np.vectorize(t_30by360)(zero_df['DATE'][:-1], zero_df['DATE'][1:])
    # Create discount df
    discount_df = pd.DataFrame(((1 + zero_df['ZERO'][1:]/2)**(-2*T_i)).values,
                               index=zero_df['DATE'][1:],
                               columns=['DISCOUNT'])
    discount_df.loc[zero_df['DATE'].iloc[0]] = 1.
    discount_df.sort_index(inplace=True)
    if not(incremental):
        # Convert discount_df to the non-incremental kind as specified in the
        # docstring
        discount_df = discount_df.cumprod(axis=0)
    return discount_df
