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
import abslibrary as lib
from numba import njit

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

def simulate_homeprice(m, spot_simulate_array, current_principal,
                       current_ltv, q=0.025, phi=0.12):
    """
    Function to simulate interest rate

    Args:
        theta_df (pd.DataFRame): value of θ(t)

        kappa, sigma (float): Hull-White parameters

    Returns:
        df containing instantaneous interest rate
    """
    home_price_array1 = spot_simulate_array.copy()*0.0
    home_price_array2 = spot_simulate_array.copy()*0.0

    deltat = 1.0/12
    row, column = spot_simulate_array.shape
    array_temp = np.random.normal(size=(row, int(column/2)))
    rand_norm = np.concatenate([array_temp, -array_temp], axis=1)

    home_price_array1[0] = current_principal[0]/current_ltv[0]
    home_price_array2[0] = current_principal[1]/current_ltv[1]

    for i in range(1, len(spot_simulate_array)):
        deltax = np.sqrt(deltat)*rand_norm[i]
        deltah = (spot_simulate_array[i-1]-q)*home_price_array1[i-1]*deltat + phi*home_price_array1[i-1]*deltax
        home_price_array1[i] = home_price_array1[i-1]+deltah

        deltah = (spot_simulate_array[i-1]-q)*home_price_array2[i-1]*deltat + phi*home_price_array2[i-1]*deltax
        home_price_array2[i] = home_price_array2[i-1]+deltah

    return home_price_array1, home_price_array2


# We need to pass in a default hazard function (parameterized), which takes an LTV parameter
# Pass in simulated HP index
# for each time step:
    # Calculate default based on previous period LTV, adjust current loan balance and home price
    # Adjust loan balance due to schedule and unscheduled principal payments
    # Construct LTV



@njit
def FRM_pool_cf(Pool_data, Pool_mwac, Pool_age, Pool_term, prepay_arr, def_haz, hp_array):
    """
    Fixed Rate Pool Cash Flow calculation. Modifies Pool_data ndarray in place

    Pool_data should be array with following 6 columns:
    0 PMT
    1 Interest
    2 Principal
    3 pp CF
    4 Balance
    5 Default

    """
    for i in range(1,Pool_term+1):
        # Calculate LTV_i
        LTV = Pool_data[i-1,4] / hp_array[i-1]
        def_t = def_haz(LTV, Pool_age+i-1)

        # reduce balance by default
        Pool_data[i,5] = def_t * Pool_data[i-1,4]
        Pool_data[i,4] = Pool_data[i-1,4]-Pool_data[i,5]
        Pool_data[i,0] = lib.pmt(Pool_data[i,4], Pool_term-i+1, Pool_mwac)
        Pool_data[i,1] = Pool_data[i,4] * Pool_mwac
        Pool_data[i,2] = ( Pool_data[i,0] - Pool_data[i,1] if (
                                            Pool_data[i,4] - Pool_data[i-1,0]
                                                + Pool_data[i-1,1] > .001  )
                                        else Pool_data[i,4])
        Pool_data[i,3] = prepay_arr[i] * (Pool_data[i-1,4] - Pool_data[i,2])
        Pool_data[i,4]  = np.maximum(Pool_data[i,4] - Pool_data[i,2] - Pool_data[i,3], 0)

    return

@njit
def ARM_pool_cf(Pool_data, rates_arr, sprd, Pool_age, Pool_term, prepay_arr, def_haz, hp_array):
    """
    Fixed Rate Pool Cash Flow calculation. Modifies Pool_data ndarray in place

    Pool_data should be array with following 6 columns:
    0 PMT
    1 Interest
    2 Principal
    3 pp CF
    4 Balance
    5 Default

    """
    for i in range(1,Pool_term+1):
        # Calculate LTV_i
        LTV = Pool_data[i-1,4] / hp_array[i-1]
        def_t = def_haz(LTV, Pool_age+i-1)

        # reduce balance by default
        Pool_data[i,5] = def_t * Pool_data[i-1,4]
        Pool_data[i,4] = Pool_data[i-1,4]-Pool_data[i,5]
        Pool_data[i,0] = lib.pmt(Pool_data[i,4], Pool_term-i+1, rates_arr[i-1] + sprd)
        Pool_data[i,1] = Pool_data[i,4] * (rates_arr[i-1] + sprd)
        Pool_data[i,2] = ( Pool_data[i,0] - Pool_data[i,1] if (
                                            Pool_data[i,4] - Pool_data[i-1,0]
                                                + Pool_data[i-1,1] > .001  )
                                        else Pool_data[i,4])
        Pool_data[i,3] = prepay_arr[i] * (Pool_data[i-1,4] - Pool_data[i,2])
        Pool_data[i,4]  = np.maximum(Pool_data[i,4] - Pool_data[i,2] - Pool_data[i,3], 0)

    return

@njit
def risky_tranche_CF_calc(tranche_CF_arr, sprd_arr, pool_CF_arr, rates_arr, orig_bal):
    """
    Calculate Tranche cash flows in place (Tranche_CF_arr ndarray)

    sprd_arr holds the spread over libor for each Tranche

    Pool_CF_arr holds the aggregated pool cash flow data
    0 PMT
    1 Interest
    2 Principal
    3 pp CF
    4 Balance
    5 Default
    """
    for i in range(1,360):

        # Calculate planned tranche data
        tranche_interest=0
        tranche_bal=0

        for j in range(tranche_CF_arr.shape[0]):
            tranche_interest = tranche_interest + tranche_CF_arr[j, i-1, 4] * (rates_arr[i-1]+sprd_arr[j])
            tranche_bal = tranche_bal + tranche_CF_arr[j,i-1,4]

            # calculate planned Interest
            tranche_CF_arr[j, i, 1] = tranche_CF_arr[j, i-1, 4] * (rates_arr[i-1]+sprd_arr[j])

        # Calculate extra principal distributions and prepayment
        excess_spread = np.maximum(pool_CF_arr[1,i] - tranche_interest, 0)
        OC = np.maximum(pool_CF_arr[4,i-1] - tranche_bal, 0)
        OC_target = np.maximum( np.minimum(0.062 * pool_CF_arr[4,i], 0.031 * orig_bal ), 3967158)
        epd = np.minimum(excess_spread, np.maximum(OC_target-OC, 0))


        # allocate principal (and unplanned principal) and interest
        available_princ = pool_CF_arr[i,2] + pool_CF_arr[i,3] + pool_CF_arr[i,5]*.4 + epd
        available_int = pool_CF_arr[i,1]

        # compute cash flow shortfall, allocate planned principal and interest, allocate PPT
        for j in range(tranche_CF_arr.shape[0]):

            # interest shortfall
            tranche_CF_arr[j, i, 5] = np.maximum( tranche_CF_arr[j, i, 1] - available_int, 0)

            # interest allocation
            tranche_CF_arr[j, i, 1] = np.minimum( available_int, tranche_CF_arr[j, i, 1])
            available_int = np.maximum(available_int - tranche_CF_arr[j,i,1],0)

            # principal
            tranche_CF_arr[j, i, 2] = np.minimum( available_princ, tranche_CF_arr[j, i-1, 4])
            available_princ = np.maximum(available_princ - tranche_CF_arr[j, i, 2], 0)

            # reduce balance
            tranche_CF_arr[j, i, 4] = np.maximum(tranche_CF_arr[j, i-1, 4] - tranche_CF_arr[j, i, 2], 0)


        # allocate default
        # eat through overcollateralization
        default_total = pool_CF_arr[i,5] - np.maximum(pool_CF_arr[i,4] - tranche_CF_arr[:, i-1, 4].sum(), 0)

        for j in reversed(range(tranche_CF_arr.shape[0])):
            default_amt = np.minimum( default_total, tranche_CF_arr[j, i, 4])
            tranche_CF_arr[j, i, 4] = np.maximum(tranche_CF_arr[j, i, 4] - default_amt, 0)
            default_total = default_total - default_amt
            tranche_CF_arr[j, i, 5] = default_amt

    return

@njit
def calc_bond_price(cf_bond, _r):
    r = _r.copy()
    #r = r.astype(float)
    r = r[:cf_bond.shape[0]]

    R = (cf_bond.sum(1)*(np.exp(r/24)-1))

    r_cum = r.cumsum()

    price = np.zeros(cf_bond.shape[1]+1)
    for i in range(price.shape[0]-1):
        price[i] = (cf_bond[:,i]/np.exp(r_cum/12)).sum()
    price[i+1] = (R/np.exp(r_cum/12)).sum()

    return price

@njit
def calc_cashflow(SMM_array_frm, SMM_array_arm, r, hpi_1, hpi_2, Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term,
                  Pool2_sprd, Pool2_age, Pool2_term, sprd_arr, Tranche_bal_arr, orig_bal, def_haz1, def_haz2):
    """
    # Pool cash flows arrays
    # 5 columns:
    #0 PMT
    0 PMT
    1 Interest
    2 Principal
    3 pp CF
    4 Balance
    5 Default
    ### Define Tranche CF arrays outside to pre-allocate
    # define Tranche CF array, Tranche x time x (Principal, Interest, Balance)
    # Tranche order: 'CG', 'VE', 'CM', 'GZ', 'TC', 'CZ', 'CA', 'CY'
    # Tranche order:    0,    1,    2,    3,    4,    5,    6,    7
    """
    Tranche_CF_arr = np.zeros((10, 361, 5))
    # 0 PMT
    # 1 Interest
    # 2 Principal
    # 3 Balance
    # 4 defaulted balance
    # 5 Interest shortfall

    for i in range(10):
        Tranche_CF_arr[i,0,2] = Tranche_bal_arr[i]

    Pool1_data = np.zeros((361, 6))
    Pool2_data = np.zeros((361, 6))

    Pool1_data[0, 4] = Pool1_bal
    Pool2_data[0, 4] = Pool2_bal

    FRM_pool_cf(Pool1_data, r, Pool1_mwac, Pool1_age, Pool1_term, SMM_array_frm, def_haz1, hpi_1)
    ARM_pool_cf(Pool2_data, r, Pool2_sprd, Pool2_age, Pool2_term, SMM_array_arm, def_haz2, hpi_2)

    # Reproduce Principal CF Allocation (waterfall)
    Total_pool_data = Pool1_data + Pool2_data

    # Calculate Cash flows
    # redefine giant Tranche_DF
    risky_tranche_CF_calc(Tranche_CF_arr, sprd_arr, Total_pool_data, r, orig_bal)

    CF_arr = np.zeros((361, 10))
    for i in range(8):
        CF_arr[:,i] = Tranche_CF_arr[i,:,1] + Tranche_CF_arr[i,:,2]

    # Calculate Bond price
    price = calc_bond_price(CF_arr[1:, :], r)
    return price

def mc_bond(m, theta_df, kappa, sigma, sol_arm_p, sol_arm_d, sol_frm_p, sol_frm_d,
            r0, tenor, antithetic, Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, sprd,
            Pool2_age, Pool2_term, sprd_arr,current_principal,current_ltv, orig_bal, Tranche_bal_arr):
    """
    current_principal current_ltv array[2]

    gamma, p: 2D arrays for hazards. (FRM, ARM) x (prepay, default)
    beta    : 2x2x2 array for covariate coefficients for hazards.
                (FRM, ARM) x (prepay, default) x ()
                [0,:] is for prepay, [1,0] is for default (LTV)
    """

    spot_simulate_df = lib.simulate_rate(m, theta_df, kappa, sigma, r0, antithetic).astype(float)
    hp_array1, hp_array2 = simulate_homeprice(m, spot_simulate_df.values, current_principal,current_ltv)

    # Calculate 10 yr rates
    tenor_rate = lib_2.calc_tenor_rate(spot_simulate_df, kappa, sigma, theta_df, tenor)

    v_frm_prepay1 = Pool1_mwac-tenor_rate
    v_frm_prepay2 = v_frm_prepay1.copy()*0.0
    v_frm_prepay2[(v_frm_prepay2.index.month>=5)&(v_frm_prepay2.index.month<=8)] = 1

    v_arm_prepay1 = sprd-tenor_rate+spot_simulate_df
    v_arm_prepay2 = v_arm_prepay1.copy()*0.0
    v_arm_prepay2[(v_arm_prepay2.index.month>=5)&(v_arm_prepay2.index.month<=8)] = 1

    smm_frm_df = lib_2.calc_hazard(sol_frm_p[0], sol_frm_p[1], sol_frm_p[2:], v_frm_prepay1, v_frm_prepay2, age=39).astype(float) # prepay hazard
    smm_arm_df = lib_2.calc_hazard(sol_arm_p[0], sol_arm_p[1], sol_arm_p[2:], v_arm_prepay1, v_arm_prepay2, age=39).astype(float) # prepay hazard
    def_haz1 = lambda LTV, t: lib_2.calc_def_hazard(sol_arm_d[0], sol_arm_d[1], sol_arm_d[2:], LTV, t) # default hazard
    def_haz2 = lambda LTV, t: lib_2.calc_def_hazard(sol_frm_d[0], sol_frm_d[1], sol_frm_d[2:], LTV, t) # default hazard


    # function definition
    #def calc_cashflow(SMM_array_arm, SMM_array_frm, r, hpi_1, hpi_2, Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term,
    #              Pool2_sprd, Pool2_age, Pool2_term, sprd_arr, Tranche_bal_arr, orig_bal, def_haz):


    price_df = np.vectorize(calc_cashflow, signature='(n),(n),(n),(n),(n),(),(),(),(),(),(),(),(),(k),(k),(),(),()->(m)')(
                            smm_frm_df.T.values, smm_arm_df.T.values,
                            spot_simulate_df.T.values,hp_array1.T,
                            hp_array2.T,Pool1_bal, Pool2_bal, Pool1_mwac,
                            Pool1_age, Pool1_term, sprd, Pool2_age, Pool2_term, sprd_arr,
                            Tranche_bal_arr, orig_bal, def_haz1, def_haz2)

    price_df = pd.DataFrame(price_df, columns=range(len(Tranche_bal_arr)))

    return price_df


@njit
def cds_valuation(tranche_CF_arr, sprd_arr, rates_arr, mat):
    """
    Calculates CDS value along a rate path (and implicit HP path)

    tranche_CF_arr
    # 0 PMT
    # 1 Interest
    # 2 Principal
    # 3 Balance
    # 4 defaulted balance
    # 5 Interest shortfall
    """

    for i in range(1,mat+1):
        # discount the default balance *.6
        # discount the interest cashflow shortfall
        pass
    return

# CDS
# 2 legs:
    # Fixed leg on changing notional (due to prepay)
    # Floating leg:
        # Default is 60% prepay
        # missing interest: Take defaulted running total, multiply by interest rate+sprd
    # add up, discount


# Fixed coupon on changing notional (discount).
    # Calculate principal and interest payments (PMT) using floating interest rate.
    # Need Tranche balance each period
    # Need Default allocation each month
# Calculate full payments, and Principal and Interest shortfall. Calculate fair payment
# Too expensive. Counterparty risk


