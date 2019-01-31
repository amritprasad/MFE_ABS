"""
MFE 230M
Library of functions
"""

# Imports
import pandas as pd
import numpy as np


def t_dattime(d0, d1, convention):
    """
    Function to calculate time in annual terms given convention

    Args:
        d0, d1: date formats that can be converted to pd.Timestamp

        convention (str): '30by360', 'ACTby360'

    Returns:
        float
    """
    # Convert to pd.Timestamp
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    if convention == '30by360':
        days_30by360 = 360*(d1.year - d0.year) + 30*(d1.month - d0.month) +\
            (d1.day - d0.day)
        return days_30by360/360
    elif convention == 'ACTby360':
        days_ACT = (d1 - d0).days
        return days_ACT/360
    elif convention == 'ACTby365':
        days_ACT = (d1 - d0).days
        return days_ACT/365
    else:
        raise ValueError('%s convention has not been implemented' % convention)


def discount_fac(_zero_df):
    """
    Function to calculate the discount rates given the zero rates. Columns
    should be DATE and ZERO

    Args:
        _zero_df (pd.DataFrame)

    Returns:
        pd.DataFrame containing discount factors between index_t and index_0.
        Index is DATE and DISCOUNT contains discount factors
    """
    # Create deep copy to avoid modifying the original df passed
    zero_df = _zero_df.copy()
    # Calculate T_i
    T_i = np.vectorize(t_dattime)(zero_df['DATE'].min(), zero_df['DATE'],
                                  '30by360')
    # Create discount df
    discount_df = pd.DataFrame(((1 + zero_df['ZERO']/2)**(-2*T_i)).values,
                               index=zero_df['DATE'],
                               columns=['DISCOUNT'])
    return discount_df


def fwd_rates(_discount_df):
    """
    Function to calculate the fwd rates given discount_df

    Args:
        _discount_df (pd.DataFrame)

    Returns:
        pd.DataFrame containing the fwd rates between index_t and index_t+1.
        Index is DATE and FWD_RATE contains the relevant fwd rate
    """
    # Create deep copy to avoid modifying the original df passed
    discount_df = _discount_df.copy()
    # Calculate T_i
    mat_dates = discount_df.index.get_level_values('DATE')
    T_i = np.vectorize(t_dattime)(mat_dates.min(), mat_dates, 'ACTby360')
    # Calculate forward rates
    tau_i = pd.Series(T_i).diff()[1:]
    fwd_df = (discount_df.divide(discount_df.shift(-1))[:-1] - 1).divide(
        tau_i.values, axis=0)
    fwd_df.columns = ['FWD_RATE']
    return fwd_df


def capswap_rates(_discount_df):
    """
    Function to calculate the cap swap rates given the discount_df

    Args:
        _discount_df (pd.DataFrame)

    Returns:
        pd.DataFrame containing swap rates of the cap
    """
    discount_df = _discount_df.copy()
    # The forward rates are paid at the end of each period. So shift the fwd_df
    # by 1 index
    fwd_df = fwd_rates(discount_df).shift(1)
    mat_dates = discount_df.index.get_level_values('DATE')
    T_i = np.vectorize(t_dattime)(mat_dates.min(), mat_dates, 'ACTby360')
    tau_i = pd.Series(T_i).diff()[1:]
    denom = tau_i.multiply(discount_df.reset_index()['DISCOUNT']).loc[2:]
    numer = tau_i.multiply(discount_df.reset_index()['DISCOUNT']).multiply(
            fwd_df.reset_index()['FWD_RATE']).loc[2:]
    capswap_df = pd.DataFrame(np.cumsum(numer)/np.cumsum(denom),
                              columns=['CAPSWAP'])
    capswap_df.index = discount_df.index[2:]
    capswap_df.dropna(inplace=True)
    capswap_df.loc[discount_df.index[1]] = fwd_df['FWD_RATE'].iloc[1]
    capswap_df.sort_index(inplace=True)
    return capswap_df


def black_price_caps(value_date, flat_vol, strike, caplet_expiry,
                     swap_pay_dates, _discount_df, notional=1e7):
    """
    Function to derive the Black price of a cap given its valuation date,
    flat vol, strike, expiry dates of the constituent caplets, pay dates of the
    underlying swap and the discount_df

    Args:
        value_date (pd.Timestamp)

        flat_vol (float)

        strike (float)

        caplet_expiry (iterable of dates)

        swap_pay_dates (iterable of dates)

        _discount_df (pd.DataFrame)

        notional (float): default value given is 10 MM

    Returns:
        Black price of cap
    """
    from scipy.stats import norm
    discount_df = _discount_df.copy()
    # The forward rates are paid at the end of each period. So shift the fwd_df
    # by 1 index. Keep only the swap_pay_dates and ignore the omitted caplet
    fwd_df = fwd_rates(discount_df).shift(1).reindex(swap_pay_dates)[1:]
    # Calculate tau_i
    tau_i = np.vectorize(t_dattime)(swap_pay_dates[:-1], swap_pay_dates[1:],
                                    'ACTby360')
    # Calculate caplet attributes
    caplet_t_i = np.vectorize(t_dattime)(value_date, caplet_expiry, 'ACTby365')
    caplet_flat_vol = flat_vol*np.sqrt(caplet_t_i)
    ln_fbyx = np.log(fwd_df/strike)
    d1 = (ln_fbyx.values.ravel() + 0.5*caplet_flat_vol**2)/caplet_flat_vol
    d2 = d1 - caplet_flat_vol
    phi1 = norm.cdf(d1)
    phi2 = norm.cdf(d2)
    cashflows = notional*tau_i*(fwd_df.values.ravel()*phi1 - strike*phi2)
    black_price = sum(discount_df.reindex(swap_pay_dates)[
            1:].values.ravel()*cashflows)
    return black_price
