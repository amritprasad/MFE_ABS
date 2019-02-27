"""
MFE 230M
Library of functions (HW 3)
"""
# Imports
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from scipy.optimize import minimize, basinhopping
import toolz
import numdifftools as nd

# Import library
import abslibrary_2 as lib_2  # HW 2 library


def good_bday_offset(dates, gbdays=-2):
    """
    Function to create DatetimeIndex after offsetting gbdays number of good
    business days

    Args:
        dates (pd.DatetimeIndex)

        gbdays (int)

    Returns:
        DatetimeIndex with good business days
    """
    # Import US Federal Holidays calendar
    cal = calendar()
    # Get list of holidays
    holidays = cal.holidays(start=dates.min(), end=dates.max())
    # Offset business days after excluding holidays
    new_dates = dates + pd.offsets.CustomBusinessDay(n=gbdays,
                                                     holidays=holidays)
    return new_dates


def fit_hazard(data_df, prepay=True, filt=False, guess=np.zeros(4),
               numdiff=True):
    """
    Function to fit hazard rate to the data

    Args:
        data_df (pd.DataFrame): contains the underlying mortgages

        prepay (bool): specify if the fitting needs to be done on the prepaid
        or defaulted mortgages

        filt (bool): specify whether defaults/prepayments need to be filtered
        out

        guess (np.1darray)

    Returns:
        scipy.optimize.OptimizeResult object
    """
    # prepay=True; filt=False; numdiff=True
    # prepay=False; filt=False; numdiff=True
    df = data_df.copy()
    covar_cols = ['Spread', 'spring_summer'] if prepay else ['LTV']
    # Convert the percentage covariates to decimals
    per_cols = ['Spread']
    df[per_cols] /= 100
    event_col = 'Prepayment_indicator' if prepay else 'Default_indicator'
    if filt:
        # Filter out data according to prepay
        remove_event = 'Default_indicator' if prepay\
            else 'Prepayment_indicator'
        remove_event_id = data_df.groupby('Loan_id')[remove_event].sum() > 0
        remove_event_id = remove_event_id[remove_event_id].index
        df = data_df[~data_df['Loan_id'].isin(remove_event_id)]
        print('%d %s filtered out' % (
                data_df.shape[0] - df.shape[0], remove_event.replace(
                        '_indicator', 's').lower()))
    covars = df[covar_cols].values

    # Fit hazard rate
    param = guess
    tb = df['period_beginning'].values
    te = df['Loan_age'].values
    event = df[event_col].values

    eps = np.finfo(float).eps
    bounds = [(eps, None), (eps, None)] + [(None, None)]*len(covar_cols)

    # Run optimizer
    # Below is the code for basinhopping which has been commented. Running it
    # requires upto 40 min. It was run independently for each of the cases and
    # the returned parameters used as the guess values for minimize
#    res_temp = basinhopping(func=lib_2.log_log_like, x0=param,
#                            minimizer_kwargs={'args': (tb, te, event, covars),
#                                              'method': 'TNC',
#                                              'jac': lib_2.log_log_grad,
#                                              'bounds': bounds})
    if numdiff:
        res_haz = minimize(fun=lib_2.log_log_like, x0=param, args=(
                tb, te, event, covars), jac=lib_2.log_log_grad, bounds=bounds,
                           method='TNC', options={'disp': True})
        if not res_haz.success:
            raise ValueError('Optimizer did not converge')
        H = nd.Hessian(lib_2.log_log_like, step=1e-4)(res_haz.x, tb, te, event,
                                                      covars)
        # Calculate the Moore-Penrose inverse since it's robust to small
        # singular values
        hessian_inv = np.linalg.pinv(H)
    else:
        res_haz = minimize(fun=lib_2.log_log_like, x0=param, args=(
                tb, te, event, covars), jac=lib_2.log_log_grad, bounds=bounds,
                           method='L-BFGS-B', options={'disp': True})
        if not res_haz.success:
            raise ValueError('Optimizer did not converge')

        hessian_inv = res_haz.hess_inv.todense()

    sol = res_haz.x
    gamma, p = sol[:2]
    beta = sol[2:]
    # Calculate standard errors as the square root of the diagonal elements
    # of the Hessian inverse
    N = len(covars)
    std_err = toolz.pipe(hessian_inv/N, np.diag, np.sqrt)
    prop_std_err = (100*std_err/abs(sol))

    print('Initial LLK: {:.2f}'.format(-lib_2.log_log_like(param, tb, te,
                                                           event, covars)))
    print('Final LLK: {:.2f}'.format(-lib_2.log_log_like(sol, tb, te,
                                                         event, covars)))
    print('\nParameters:')
    if 'spring_summer' in covar_cols:
        print('gamma =', gamma, '\np =', p, '\nCoupon Gap =', beta[0],
              '\nSummer Indicator =', beta[1])
    else:
        print('gamma =', gamma, '\np =', p, '\nLTV =', beta[0])
    print('\nRespective Standard Errors:', ', '.join(
            std_err.round(3).astype(str)))
    print('\nProportional Standard Errors:', ', '.join(prop_std_err.round(
            1).astype(str)))

    return res_haz, res_haz.x, hessian_inv

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
    spot_simulate_df.index = theta_df.index
    #spot_simulate_df.index = range(1, len(spot_simulate_df)+1)
    return spot_simulate_df

def simulate_homeprice(m, spot_simulate_df, current_principal,
                       current_ltv, q=0.025, phi=0.12):
    """
    Function to simulate interest rate

    Args:
        theta_df (pd.DataFRame): value of θ(t)

        kappa, sigma (float): Hull-White parameters

    Returns:
        df containing instantaneous interest rate
    """
    home_price_df = spot_simulate_df.copy()*0.0
    deltat = 1.0/12
    row, column = spot_simulate_df.shape
    df_temp = pd.DataFrame(np.random.normal(size=(row, int(column/2))))
    rand_norm = pd.concat([df_temp, -df_temp], axis=1)
    rand_norm.index = spot_simulate_df.index
    rand_norm.columns = spot_simulate_df.columns

    home_price_df.iloc[0] = current_principal/current_ltv
    for i in range(1, len(spot_simulate_df)):
        deltax = np.sqrt(deltat)*rand_norm.iloc[i]
        deltah = (spot_simulate_df.iloc[i-1]-q)*home_price_df.iloc[i-1]*deltat + phi*home_price_df.iloc[i-1]*deltax
        home_price_df.iloc[i] = home_price_df.iloc[i-1]+deltah

    return home_price_df

def mc_bond(m, theta_df, kappa, sigma, gamma, p, beta, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
            Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate):
    spot_simulate_df = simulate_rate(m, theta_df, kappa, sigma, r0, antithetic)
    home_price_df = simulate_homeprice(m, spot_simulate_df, current_principal, current_ltv)

    # Calculate 10 yr rates
    tenor_rate = calc_tenor_rate(spot_simulate_df, kappa, sigma, theta_df, tenor)

    v1 = wac-tenor_rate
    v2 = v1.copy()*0.0
    v2[(v2.index.month>=4)&(v2.index.month<=7)] = 1

    smm_df = calc_hazard(gamma, p, beta, v1, v2)

    price_df = np.vectorize(calc_cashflow, signature='(n),(n),(),(),(),(),(),(),(),(),(),(k)->(m)')(
                            smm_df.T.values.astype(float), spot_simulate_df.T.values.astype(float),
                            Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term,
                            Pool2_mwac, Pool2_age, Pool2_term, coupon_rate,Tranche_bal_arr)
    price_df = pd.DataFrame(price_df, columns=bond_list)

    return price_df, smm_df