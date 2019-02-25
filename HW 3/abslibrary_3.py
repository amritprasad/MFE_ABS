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


def fit_hazard(data_df, prepay=True, filt=False, guess=np.zeros(4)):
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
    # prepay=True; filt=False
    # prepay=False; filt=False
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
#                                              'method': 'L-BFGS-B',
#                                              'jac': lib_2.log_log_grad,
#                                              'bounds': bounds})
    res_haz = minimize(fun=lib_2.log_log_like, x0=param, args=(
            tb, te, event, covars), jac=lib_2.log_log_grad, bounds=bounds,
                       method='L-BFGS-B', options={'disp': True})
    if not res_haz.success:
        raise ValueError('Optimizer did not converge')

    sol = res_haz.x
    gamma, p = sol[:2]
    beta = sol[2:]
    # Calculate standard errors as the square root of the diagonal elements of
    # the Hessian inverse
    N = len(covars)
    hessian_inv = res_haz.hess_inv.todense()
    std_err = toolz.pipe(hessian_inv/N, np.diag, np.sqrt)
    prop_std_err = (100*std_err/sol)

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
