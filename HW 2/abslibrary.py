"""
MFE 230M
Library of functions
"""

# Imports
import pandas as pd
import numpy as np
import scipy

Tranche_bal_dict = {
                    'CG': 74800000,
                    'VE': 5200000,
                    'CM': 14000000,
                    'GZ': 22000000,
                    'TC': 20000000,
                    'CZ': 24000000,
                    'CA': 32550000,
                    'CY': 13950000
                   }


def t_dattime(d0, d1, convention):
    """
    Function to calculate time in annual terms given convention

    Args:
        d0, d1: date formats that can be converted to pd.Timestamp

        convention (str): '30by360', 'ACTby360', 'ACTby365'

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


def hullwhite_price_caps(value_date, sigma, kappa, strike, swap_pay_dates,
                         _discount_df, notional=1e7):
    """
    Function to calculate the Hull-White Cap price of a cap given its valuation
    date, sigma, kappa, strike, expiry dates of the constituent caplets,
    pay dates of the underlying swap and the discount_df

    Args:
        value_date (pd.Timestamp)

        sigma, kappa (float): Hull-White parameters

        strike (float)

        swap_pay_dates (iterable of dates)

        _discount_df (pd.DataFrame)

        notional (float): default value given is 10 MM

    Returns:
        Black price of cap
    """
    from scipy.stats import norm
    discount_df = _discount_df.copy()
    # Calculate Hull-White price
    t_i = swap_pay_dates[1:]
    t_im1 = swap_pay_dates[:-1]
    tau_i = np.vectorize(t_dattime)(t_im1, t_i, 'ACTby360')
    Delta_im1 = np.vectorize(t_dattime)(value_date, t_im1, 'ACTby365')
    Z_i = discount_df.reindex(swap_pay_dates)[1:].values.ravel()
    Z_im1 = discount_df.reindex(swap_pay_dates)[:-1].values.ravel()
    # Calculate sigma_p_i
    sigma_p_i = sigma*np.sqrt((1-np.exp(-2*kappa*Delta_im1))/2/kappa)/kappa*(
        1-np.exp(-kappa*tau_i))
    h_i = sigma_p_i/2 + np.log(Z_i*(1+strike*tau_i)/Z_im1)/sigma_p_i
    caplet_i = Z_im1*norm.cdf(-h_i+sigma_p_i) - (
            1+strike*tau_i)*Z_i*norm.cdf(-h_i)
    return caplet_i.sum()*notional


def loss_hw_black(params, black_prices, annual_idx, swap_pay_dates,
                  value_date, capatmstrike_df, discount_df, notional=1e7):
    """
    Function to calculate the difference in prices between the Hull-White model
    and the Black model

    Args:
        params (list): kappa, sigma

        black_prices (iterable): contains the Black implied cap prices

        annual_idx (iterable): contains cut-off index for caplet expiries and
        swap pay dates

        swap_pay_dates (iterable): contains the underlying swap dates

        value_date (pd.Timestamp): valuation date

        capatmstrike_df (pd.DataFrame): contains the strikes of the caps

        discount_df (pd.DataFrame): contains the discount rates

        notional (float): notional of the Black prices

    Returns:
        squared difference in prices between Hull-White and Black
    """
    kappa, sigma = params
    error_str = "Number of Cap strikes and prices don't match"
    assert capatmstrike_df.shape[0] == len(black_prices), error_str
    hw_prices = np.zeros(black_prices.size)
    for i in range(hw_prices.size):
        hw_prices[i] = hullwhite_price_caps(
                value_date, sigma, kappa, capatmstrike_df['CAPSWAP'].iloc[i],
                swap_pay_dates[:annual_idx[i]+1], discount_df,
                notional=notional)
    if np.isnan(hw_prices).any():
        return np.inf
    else:
        return sum((hw_prices - black_prices)**2)


def loss_black_hw(param, hw_price, caplet_expiry, swap_pay_dates,
                  value_date, strike, discount_df, notional=1e7):
    """
    Function to calculate the difference in prices between the Hull-White model
    and the Black model

    Args:
        param (float): flat_vol

        hw_price (iterable): Hull-White cap price

        annual_idx (iterable): contains cut-off index for caplet expiries and
        swap pay dates

        swap_pay_dates (iterable): contains the underlying swap dates

        value_date (pd.Timestamp): valuation date

        capatmstrike_df (pd.DataFrame): contains the strikes of the caps

        discount_df (pd.DataFrame): contains the discount rates

        notional (float): notional of the Black prices

    Returns:
        squared difference in prices between Hull-White and Black
    """
    flat_vol = param
    black_price_hwvol = black_price_caps(
            value_date, flat_vol, strike, caplet_expiry, swap_pay_dates,
            discount_df, notional=notional)
    if np.isnan(black_price_hwvol):
        return np.inf
    else:
        return (hw_price - black_price_hwvol)**2


def inst_f(_discount_df, time_step, derivative):
    """
    Function to estimate the instantaneous forward rates

    Args:
        _discount_df (pd.DataFrame): contains the discount rates

        time_step (int): time step in months

        derivative (bool): specify if derivative needs to be calculated

    Returns:
        df containing instantaneous forward rates
    """
    discount_df = _discount_df.copy()
    t_i = discount_df.index[1:]
    t_im1 = discount_df.index[:-1]
    delta_t = np.vectorize(t_dattime)(t_im1, t_i, 'ACTby365')
    Z_i = discount_df['DISCOUNT'][1:].values.ravel()
    Z_im1 = discount_df['DISCOUNT'][:-1].values.ravel()
    f_M_t = np.log(Z_im1/Z_i)/delta_t
    f_M_t = pd.DataFrame(f_M_t, index=discount_df.index[:-1],
                         columns=['INST_FWD_RATE'])
    # Interpolate linearly for all the intermediate time steps
    inst_fwd_df = f_M_t.resample(str(time_step)+'MS').interpolate(
            method='cubic', order=3)
    # Calculate partial derivative wrt time if specified
    if derivative:
        spacing = np.vectorize(t_dattime)(inst_fwd_df.index.min(),
                                          inst_fwd_df.index, 'ACTby365')
        dinst_fwd_df = pd.DataFrame(np.gradient(
                inst_fwd_df.values.ravel(), spacing), index=inst_fwd_df.index,
                                    columns=['del_INST_FWD_RATE'])
        return dinst_fwd_df
    return inst_fwd_df


def hw_theta(kappa, sigma, _discount_df, start_date):
    """
    Function to calculate θ(t)

    Args:
        kappa, sigma (float): Hull-White parameters

        _discount_df (pd.DataFrame): contains the discount rates and the dates
        for which θ(t) would be calculated

        start_date (pd.Timestamp)

    Returns:
        df containing θ(t)
    """
    discount_df = _discount_df.copy()
    inst_fwd_df = inst_f(discount_df, time_step=1, derivative=False)
    dinst_fwd_df = inst_f(discount_df, time_step=1, derivative=True)
    t = np.vectorize(t_dattime)(start_date, inst_fwd_df.index, 'ACTby365')
    theta_df = kappa*inst_fwd_df.values.ravel() + dinst_fwd_df.values.ravel()\
        + (sigma**2)*(1-np.exp(-2*kappa*t))/2/kappa
    theta_df = pd.DataFrame(theta_df, index=inst_fwd_df.index,
                            columns=['THETA'])
    return theta_df


def simulate_rate(m, theta_df, kappa, sigma, r0, antithetic):
    """
    Function to simulate interest rate

    Args:
        theta_df (pd.DataFRame): value of θ(t)

        kappa, sigma (float): Hull-White parameters

    Returns:
        df containing instantaneous interest rate
    """
    spot_simulate_df = theta_df.resample('1MS').first()*np.nan
    spot_simulate_df = pd.DataFrame(index=spot_simulate_df.index,
                                    columns=range(1, m+1))
    deltat = 1.0/12
    if antithetic:
        row, column = spot_simulate_df.shape
        df_temp = pd.DataFrame(np.random.normal(size=(row, int(column/2))))
        rand_norm = pd.concat([df_temp, -df_temp], axis=1)
        rand_norm.index = spot_simulate_df.index
        rand_norm.columns = spot_simulate_df.columns
    else:
        rand_norm = pd.DataFrame(np.random.normal(size=spot_simulate_df.shape),
                                 index=spot_simulate_df.index,
                                 columns=spot_simulate_df.columns)
    spot_simulate_df.iloc[0] = r0
    for i in range(1, len(spot_simulate_df)):
        deltax = np.sqrt(deltat)*rand_norm.iloc[i]
        deltar = (theta_df.iloc[i, 0]-spot_simulate_df.iloc[
                i-1]*kappa)*deltat+sigma*deltax
        spot_simulate_df.iloc[i] = spot_simulate_df.iloc[i-1]+deltar
    spot_simulate_df.index = range(1, len(spot_simulate_df)+1)
    return spot_simulate_df


def mc_bond(m, cf_bond, theta_df, kappa, sigma, r0, antithetic=False):
    r = simulate_rate(m, theta_df, kappa, sigma, r0, antithetic).astype(float)
    r = r.iloc[:len(cf_bond)]

    R = (cf_bond.sum(1).T*(np.exp(r/24)-1).T).T

    r_cum = r.cumsum()

    price_dict = {}
    for i in cf_bond.columns:
        price_dict[i] = (cf_bond[i].T/np.exp(r_cum/12).T).T.sum()
    price_dict['R'] = (R/np.exp(r_cum/12)).sum()

    price_df = pd.DataFrame(price_dict)
    if antithetic:
        length = int(m/2)
        for i in range(length):
            price_df.loc[i+1] = (price_df.loc[i+1]+price_df.loc[i+1+length])/2
        price_df = price_df.loc[range(1, length+1)]
    return price_df


def calc_duration_convexity(m, cf_bond, theta_df, kappa, sigma, r0,
                            antithetic=True):
    deltar = 0.0002
    price_pos = mc_bond(m, cf_bond, theta_df, kappa, sigma, r0+deltar,
                        antithetic).mean()
    price = mc_bond(m, cf_bond, theta_df, kappa, sigma, r0, antithetic).mean()
    price_neg = mc_bond(m, cf_bond, theta_df, kappa, sigma, r0-deltar,
                        antithetic).mean()
    duration = (price_neg-price_pos)/price/2/deltar
    convexity = (price_pos+price_neg-price*2)/price/deltar
    return duration, convexity


def calc_PV_diff(r, cf, zero_df, par):
    _zero_df = zero_df.copy()
    _zero_df['ZERO'] = _zero_df['ZERO']+r
    discount_df = discount_fac(_zero_df)
    discount_df.index = range(0, len(discount_df))
    pv = (discount_df.iloc[:, 0]*cf).sum()
    # print(pv-par)
    return pv-par


def calc_OAS(cf_bond, zero_df):
    r_dict = {}
    _zero_df = zero_df.copy()
    _zero_df.index = _zero_df['DATE']
    _zero_df = _zero_df[['ZERO']]
    _zero_df = _zero_df.resample('1MS').interpolate(method='index')
    _zero_df['DATE'] = _zero_df.index
    for column in cf_bond.columns:
        r0 = 0
        r_dict[column] = scipy.optimize.fsolve(calc_PV_diff, r0, args=(
                cf_bond[column], _zero_df, Tranche_bal_dict[column]))[0]
    r_ser = pd.Series(r_dict, index=cf_bond.columns)
    return r_ser


# %%
# ABS Cash Flow Functions
# %% Calculate Pool principal payments. The principal payments pass through
# to the bond holders. To do this we need to:
# 1) Calculate the amortizing payment to include the principal pre-payments
# 2) Subtract the interest payments to get principal payments
# 3) Multiply monthly ppmt rate by (prev bal - princ pmts) to get ppmts
# This must be done sequentially (serial dependence), so looping is quite slow.
# We may want to write numba functions to speed it up, particularly when
#   cash flows become dependent on our state variables.

# First we define helper functions

def pmt(bal, n_per, rate):
    """
    Calculates amortizing payments
    """
    if abs(bal - 0) < 1e-2:
        return 0.0

    return bal * rate * (1+rate)**n_per / ((1+rate)**n_per - 1)


# Keep separate CPR array so that we can make it depend on the short rate path
def cpr(PSA, pp_rate, total_term, age, rate_path=None):
    """
    Function to give us CPR schedule
    """
    arr = np.linspace(0, total_term-1, num=total_term, dtype=int)
    CPR_array = PSA * pp_rate * np.minimum(1, (arr+age)/30)
    return CPR_array


def pool_cf(Pool_data, Pool_mwac, Pool_age, Pool_term, SMM_array):
    """
    Fills in Pool data array
    """
    for i in range(1,Pool_term+1): # this loop can be sped up with numba
        Pool_data.loc[i,'PMT']      = pmt(Pool_data.loc[i-1,'Balance'], Pool_term-i+1, Pool_mwac)
        Pool_data.loc[i,'Interest'] = Pool_data.loc[i-1,'Balance'] * Pool_mwac
        Pool_data.loc[i,'Principal']= ( Pool_data.loc[i,'PMT'] - Pool_data.loc[i,'Interest'] if (
                                            Pool_data.loc[i-1,'Balance'] - Pool_data.loc[i-1,'PMT']
                                                + Pool_data.loc[i-1,'Interest'] > .001  )
                                        else Pool_data.loc[i-1,'Balance'])
        Pool_data.loc[i,'PP CF']    = SMM_array[i] * (Pool_data.loc[i-1,'Balance'] -
                                                  Pool_data.loc[i,'Principal'])
        Pool_data.loc[i,'Balance']  = Pool_data.loc[i-1,'Balance'] - Pool_data.loc[i,'Principal'] \
                                        - Pool_data.loc[i,'PP CF']

    return


def tranche_CF_calc(Tranche_dict, CA_CY_princ, Rest_princ, GZ_interest,
                    CZ_interest, coupon_rate):
    """
    Function to calculate Tranche waterfalls
    """
    for i in range(1,241):
        # principal to CA, then to CY
        Tranche_dict['CA'].loc[i,'Principal']   = min( CA_CY_princ[i],
                                                    Tranche_dict['CA'].loc[i-1,'Balance'] )
        Tranche_dict['CA'].loc[i,'Balance']     = ( Tranche_dict['CA'].loc[i-1,'Balance']
                                                    - Tranche_dict['CA'].loc[i,'Principal'] )
        Tranche_dict['CY'].loc[i,'Principal']   = CA_CY_princ[i] - Tranche_dict['CA'].loc[i,'Principal']
        Tranche_dict['CY'].loc[i,'Balance']     = ( Tranche_dict['CY'].loc[i-1,'Balance']
                                                    - Tranche_dict['CY'].loc[i,'Principal'] )
        Tranche_dict['CA'].loc[i,'Interest']    = Tranche_dict['CA'].loc[i-1,'Balance'] * coupon_rate
        Tranche_dict['CY'].loc[i,'Interest']    = Tranche_dict['CY'].loc[i-1,'Balance'] * coupon_rate

        # CG, VE, CM, GZ, TC, CZ
        # Accrual interest components are independent of other calculations
        GZ_interest.loc[i,'interest'] = Tranche_dict['GZ'].loc[i-1,'Balance'] * coupon_rate
        CZ_interest.loc[i,'interest'] = Tranche_dict['CZ'].loc[i-1,'Balance'] * coupon_rate

        #CG
        Tranche_dict['CG'].loc[i,'Principal'] = max( 0, min( Tranche_dict['CG'].loc[i-1,'Balance'],
                                                        Rest_princ[i] + CZ_interest.loc[i,'interest']
                                                        ))
        Tranche_dict['CG'].loc[i,'Balance'] = Tranche_dict['CG'].loc[i-1,'Balance'] \
                                                - Tranche_dict['CG'].loc[i,'Principal']
        Tranche_dict['CG'].loc[i,'Interest'] = Tranche_dict['CG'].loc[i-1,'Balance'] * coupon_rate

        #VE
        Tranche_dict['VE'].loc[i,'Principal'] = max( 0, min( Tranche_dict['VE'].loc[i-1,'Balance'],
                                                        Rest_princ[i] + CZ_interest.loc[i,'interest'] \
                                                        + GZ_interest.loc[i,'interest'] \
                                                        - Tranche_dict['CG'].loc[i,'Principal']
                                                        ))
        Tranche_dict['VE'].loc[i,'Balance'] = Tranche_dict['VE'].loc[i-1,'Balance'] \
                                                - Tranche_dict['VE'].loc[i,'Principal']
        Tranche_dict['VE'].loc[i,'Interest'] = Tranche_dict['VE'].loc[i-1,'Balance'] * coupon_rate

        #CM
        Tranche_dict['CM'].loc[i,'Principal'] = max( 0, min( Tranche_dict['CM'].loc[i-1,'Balance'],
                                                        Rest_princ[i] + CZ_interest.loc[i,'interest'] \
                                                        + GZ_interest.loc[i,'interest'] \
                                                        - Tranche_dict['CG'].loc[i,'Principal'] \
                                                        - Tranche_dict['VE'].loc[i,'Principal']
                                                        ))
        Tranche_dict['CM'].loc[i,'Balance'] = Tranche_dict['CM'].loc[i-1,'Balance'] \
                                                - Tranche_dict['CM'].loc[i,'Principal']
        Tranche_dict['CM'].loc[i,'Interest'] = Tranche_dict['CM'].loc[i-1,'Balance'] * coupon_rate

        # GZ
        if Tranche_dict['CM'].loc[i,'Balance']>1e-1:
            GZ_interest.loc[i,'accrued'] = GZ_interest.loc[i,'interest']
        else:
            GZ_interest.loc[i,'accrued'] = min( GZ_interest.loc[i,'interest'],
                                                       Tranche_dict['TC'].loc[i,'Principal'])
        Tranche_dict['GZ'].loc[i,'Principal'] = max( 0, min( Tranche_dict['GZ'].loc[i-1,'Balance'],
                                                        Rest_princ[i] + CZ_interest.loc[i,'interest'] \
                                                        + GZ_interest.loc[i,'accrued'] \
                                                        - Tranche_dict['CG'].loc[i,'Principal'] \
                                                        - Tranche_dict['VE'].loc[i,'Principal'] \
                                                        - Tranche_dict['CM'].loc[i,'Principal']
                                                        ))
        Tranche_dict['GZ'].loc[i,'Balance'] = Tranche_dict['GZ'].loc[i-1,'Balance'] \
                                                + GZ_interest.loc[i,'accrued'] \
                                                - Tranche_dict['GZ'].loc[i,'Principal']
        Tranche_dict['GZ'].loc[i,'Interest'] = GZ_interest.loc[i,'interest'] \
                                                - GZ_interest.loc[i,'accrued']

        #TC
        Tranche_dict['TC'].loc[i,'Principal'] = 0 if Tranche_dict['GZ'].loc[i,'Balance'] > 1e-1 else (
                                                        min( Tranche_dict['TC'].loc[i-1,'Balance'],
                                                        Rest_princ[i] + CZ_interest.loc[i,'interest'] \
                                                        - Tranche_dict['GZ'].loc[i,'Principal']
                                                        ))
        Tranche_dict['TC'].loc[i,'Balance'] = Tranche_dict['TC'].loc[i-1,'Balance'] \
                                                - Tranche_dict['TC'].loc[i,'Principal']
        Tranche_dict['TC'].loc[i,'Interest'] = Tranche_dict['TC'].loc[i-1,'Balance'] * coupon_rate

        # CZ
        if Tranche_dict['TC'].loc[i,'Balance']>1e-1:
            CZ_interest.loc[i,'accrued'] = CZ_interest.loc[i,'interest']
        else:
            CZ_interest.loc[i,'accrued'] = min( CZ_interest.loc[i,'interest'],
                                                       Tranche_dict['TC'].loc[i,'Principal'])
        Tranche_dict['CZ'].loc[i,'Principal'] = max( 0, min( Tranche_dict['CZ'].loc[i-1,'Balance'],
                                                        Rest_princ[i] + CZ_interest.loc[i,'accrued'] \
                                                        - Tranche_dict['CG'].loc[i,'Principal'] \
                                                        - Tranche_dict['VE'].loc[i,'Principal'] \
                                                        - Tranche_dict['CM'].loc[i,'Principal'] \
                                                        - Tranche_dict['GZ'].loc[i,'Principal'] \
                                                        - Tranche_dict['TC'].loc[i,'Principal']
                                                        ))
        Tranche_dict['CZ'].loc[i,'Balance'] = Tranche_dict['CZ'].loc[i-1,'Balance'] \
                                                + CZ_interest.loc[i,'accrued'] \
                                                - Tranche_dict['CZ'].loc[i,'Principal']
        Tranche_dict['CZ'].loc[i,'Interest'] = CZ_interest.loc[i,'interest'] \
                                                - CZ_interest.loc[i,'accrued']
