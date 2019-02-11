"""
MFE 230M
Library of functions (HW 2)
"""
# Imports
import pandas as pd
import numpy as np
import scipy

# Import library
import abslibrary as lib  # HW 1 library
from numba import njit

phist = []
gradhist = []
cnt = 0


def log_log_grad(param, tb, te, event, covars):
    """
    This function calculates the gradient of the log-likelihood for the
    proportional hazard model using the log-logistics baseline distribution
    """
    g = param[0]  # Amplitude of the baseline hazard; gamma in the notation
    p = param[1]  # Shape of baseline hazard; p in the notation
    coef = param[2:]  # Coefficients for covariates; beta in the notation

    dlldg1 = sum(event*(p/g-(p*g**(p-1)*(te**p))/(1+(g*te)**p)))
    if len(covars):
        dlldg2 = sum((p*g**(p-1))*((te**p/(1+(g*te)**p))-(tb**p/(1+(g*tb)**p)))*np.exp(covars.dot(coef)))
    else:
        dlldg2 = sum((p*g**(p-1))*((te**p/(1+(g*te)**p))-(tb**p/(1+(g*tb)**p))))

    dlldg = -(dlldg1-dlldg2)

    dlldp1 = sum(event*(1/p+np.log(g*te)-(g*te)**p*np.log(g*te)/(1+(g*te)**p)))

    # When tb = 0, calculate the derivative of the unconditional survival
    # function. This is because the derivative of the conditional survival
    # function does not generalize to the unconditional case when tb = 0. There
    # is a singularity on log(g*tb) for tb = 0.

    mask = tb == 0
    ln_gtb = np.empty(tb.size)
    ln_gtb[mask] = 0
    ln_gtb[~mask] = np.log((g*tb)[~mask])
    ln_gtb[np.isposinf(ln_gtb)] = 0
    ln_gtb[np.isneginf(ln_gtb)] = 0

    if len(covars):
        dlldp2 = sum((((g*te)**p)*np.log(g*te)/(1+(g*te)**p)-(g*te)**p*ln_gtb/(1+(g*tb)**p))*np.exp(covars.dot(coef)))
    else:
        dlldp2 = sum((((g*te)**p)*np.log(g*te)/(1+(g*te)**p)-(g*te)**p*ln_gtb/(1+(g*tb)**p)))

    dlldp = -(dlldp1-dlldp2)

    grad = [dlldg, dlldp]

    for i in range(0, len(coef)):
        dlldc1 = sum(event*covars[:, i])
        dlldc2 = sum((np.log(1+(g*te)**p)-np.log(1+(g*tb)**p))*np.exp(
                covars.dot(coef))*covars[:, i])
        dlldc = -(dlldc1-dlldc2)

        grad.append(dlldc)

    grad = np.array(grad)

    return grad


def log_log_like(param, tb, te, event, covars):
    """
    This function calculates the log likelihood for a proportional hazard
    model with log-logistic baseline hazard.  It can be used to solve for
    the parameters of the model.
    """
    # tb=static_df['period_begin']/365;te=static_df['period_end']/365
    # event=static_df['prepay'];param=[0.1]*7

    # Get the number of parameters
    nentries = len(te)

    g = param[0]  # Amplitude of the baseline hazard; gamma in the notation
    p = param[1]  # Shape of baseline hazard; p in the notation
    coef = param[2:]  # Coefficients for covariates; beta in the notation

    # The following variables are vectors with a row for each episode
    # Log of baseline hazard
    logh = (np.log(p) + np.log(g) + (p-1)*(np.log(g)+np.log(te))
            - np.log(1+(g*te)**p))

    logc = np.zeros(nentries)
    logF = -(np.log(1+(g*te)**p) - np.log(1+(g*tb)**p))
    if len(covars):
        # Product of covarites and coefficients
        logc = (covars.dot(coef)).flatten()
        # Log of conditional survival function
        logF = logF*np.exp(covars.dot(coef))

    # Construct the negative of log likelihood
    neglogL = -(sum(event*(logh+logc)) + sum(logF))

    # Calculate the derivative of the log likelihood with respect to each
    # parameter. In order for the maximum likelihood estimation to converge it
    # is necessary to provide these derivatives so that the search algogrithm
    # knows which direction to search in.

    global phist, cnt, gradhist

    grad = log_log_grad(param, tb, te, event, covars)
    gradhist.append(grad)
    phist.append(param)

    cnt += 1
    if cnt % 100 == 0:
        print('%d evaluations of negative LLK function completed' % cnt)

    return neglogL


def hw_B(kappa, delta_t):
    """
    Function to calculate B(t, T) according to the Hull-White model.
    B(t, T) = B(0, T-t)

    Args:
        kappa (float)

        delta_t (float): in years

    Returns:
        float B
    """
    B = (1 - np.exp(-kappa*delta_t))/kappa
    return B


def hw_A(kappa, sigma, theta, tenor):
    """
    Function to calculate A(t, T) according to the Hull-White model.
    A(t, T) = A(0, T-t)

    Args:
        kappa, sigma (float)

        theta (pd.DataFrame)

        tenor (int): in months

    Returns:
        pd.Series containing values for A for each month
    """
    # theta=theta_df.copy(); tenor=120
    # Convert theta index from dates to numerical to speed up estimation
    old_index = theta.index
    index = np.vectorize(lib.t_dattime)(old_index.min(), old_index, 'ACTby365')
    theta.index = index
    theta.reset_index(inplace=True)

    index_diff = theta['index'].diff().shift(-1)
    cumul_index_diff = index_diff.rolling(tenor).sum().shift(-tenor+1)
    B = np.vectorize(hw_B)(kappa, index_diff)
    integ = theta['THETA']*B
    integ = integ.rolling(tenor).sum()
    integ = integ.shift(-tenor+1)

    B_tenor = np.vectorize(hw_B)(kappa, cumul_index_diff)
    non_integ = 0.5*(sigma/kappa)**2*(
                cumul_index_diff + (1-np.exp(
                        -2*kappa*cumul_index_diff))/2/kappa - 2*B_tenor)

    A = pd.DataFrame(non_integ - integ, columns=['A'])
    A['DATE'] = theta['index']
    A['DATE'] = A['DATE'].map({key: val for key, val in zip(index, old_index)})
    A.set_index('DATE', inplace=True)
    return A['A']


def calc_tenor_rate(spot_simulate_df, kappa, sigma, theta, tenor):
    """
    Function to calculate the rates corresponding to the tenor

    Args:
        spot_simulate_df (pd.DataFrame): contains simulated spot rate

        kappa: float

        sigma: float

        theta(pd.DataFrame)

        tenor (int): in months

    Returns:
        pd.DataFrame containing tenor rates
    """
    _theta = theta.copy()
    _spot_simulate_df = spot_simulate_df.copy()
    _spot_simulate_df.index = theta.index
    A = hw_A(kappa, sigma, _theta, tenor)
    part1 = -A/tenor
    part2 = 1/kappa*(1-np.exp(-kappa*tenor))/tenor*_spot_simulate_df

    return (part1 + part2.T).T


def calc_hazard(gamma, p, beta, v1, v2):
    """
    Function to give us CPR schedule
    """
    _v1 = v1.copy()
    _v1.index = range(len(v1))
    _v2 = v2.copy()
    _v2.index = range(len(v2))
    part1 = pd.Series((gamma*p)*(gamma*_v1.index)**(p-1)/(1+(gamma*_v1.index)**p), index = _v1.index)
    part2 = beta[0]*_v1+beta[1]*_v2
    part3 = np.exp(part2.astype(float))
    hazard_rate = (part1*part3.T).T
    hazard_rate.index = v1.index
    return hazard_rate

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
def calc_cashflow(SMM_array, r, Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term,
                  Pool2_mwac, Pool2_age, Pool2_term, coupon_rate, Tranche_bal_arr):
    # Pool cash flows arrays
    # 5 columns:
    #0 PMT
    #1 Interest
    #2 Principal
    #3 pp CF
    #4 Balance

    ### Define Tranche CF arrays outside to pre-allocate
    # define Tranche CF array, Tranche x time x (Principal, Interest, Balance)
    # Tranche order: 'CG', 'VE', 'CM', 'GZ', 'TC', 'CZ', 'CA', 'CY'
    # Tranche order:    0,    1,    2,    3,    4,    5,    6,    7
    Tranche_CF_arr = np.zeros((8,241,3))
    for i in range(8):
        Tranche_CF_arr[i,0,2] = Tranche_bal_arr[i]


    #a=time.time()
    Pool1_data = np.zeros((241, 5))
    Pool2_data = np.zeros((241, 5))

    Pool1_data[0, 4] = Pool1_bal
    Pool2_data[0, 4] = Pool2_bal

    lib.pool_cf(Pool1_data, Pool1_mwac, Pool1_age, Pool1_term, SMM_array)
    lib.pool_cf(Pool2_data, Pool2_mwac, Pool2_age, Pool2_term, SMM_array)

    # Reproduce Principal CF Allocation (waterfall)
    Total_principal = Pool1_data[:,2] + Pool2_data[
            :,2] + Pool1_data[:,3] + Pool2_data[:,3]
    CA_CY_princ = 0.225181598 * Total_principal
    Rest_princ = 0.7748184 * Total_principal

    # Temporary arrays for calculating GZ/CZ interest accrual. The "interest" here
    # is not cash flow. 2 columns:
    #interest
    #accrued
    GZ_interest = np.zeros((241, 2))
    CZ_interest = np.zeros((241, 2))

    # Calculate Cash flows
    # redefine giant Tranche_DF
    lib.tranche_CF_calc(Tranche_CF_arr, CA_CY_princ, Rest_princ, GZ_interest,
                        CZ_interest, coupon_rate)

    CF_arr = np.zeros((241, 8))
    for i in range(8):
        CF_arr[:,i] = Tranche_CF_arr[i,:,0] + Tranche_CF_arr[i,:,1]

    #print(time.time()-a)
    # Calculate Bond price
    price = calc_bond_price(CF_arr[1:, :], r)
    return price

def mc_bond(m, theta_df, kappa, sigma, gamma, p, beta, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
            Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate):
    spot_simulate_df = lib.simulate_rate(m, theta_df, kappa, sigma, r0, antithetic)

    # Calculate 10 yr rates
    tenor_rate = calc_tenor_rate(spot_simulate_df, kappa, sigma, theta_df, tenor)

    v1 = wac-tenor_rate
    v2 = v1.copy()*0.0
    v2[(v2.index.month>=5)&(v2.index.month<=8)] = 1

    smm_df = calc_hazard(gamma, p, beta, v1, v2)

    price_df = np.vectorize(calc_cashflow, signature='(n),(n),(),(),(),(),(),(),(),(),(),(k)->(m)')(
                            smm_df.T.values.astype(float), spot_simulate_df.T.values.astype(float),
                            Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term,
                            Pool2_mwac, Pool2_age, Pool2_term, coupon_rate,Tranche_bal_arr)
    price_df = pd.DataFrame(price_df, columns=bond_list)

    return price_df

def calc_duration_convexity(m, theta_df, kappa, sigma, gamma, p, beta, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                            Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate):
    deltar = 0.0025
    price_pos = mc_bond(m, theta_df, kappa, sigma, gamma, p, beta, r0+deltar, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                        Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate).mean()
    price = mc_bond(m, theta_df, kappa, sigma, gamma, p, beta, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                    Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate).mean()
    price_neg = mc_bond(m, theta_df, kappa, sigma, gamma, p, beta, r0-deltar, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
                        Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate).mean()
    duration = (price_neg-price_pos)/price/2/deltar
    convexity = (price_pos+price_neg-price*2)/price/deltar**2
    return duration, convexity

def calc_cashflow_CA(SMM_array, r, Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term,
                     Pool2_mwac, Pool2_age, Pool2_term, coupon_rate, Tranche_bal_arr):
    # Pool cash flows arrays
    # 5 columns:
    #0 PMT
    #1 Interest
    #2 Principal
    #3 pp CF
    #4 Balance

    ### Define Tranche CF arrays outside to pre-allocate
    # define Tranche CF array, Tranche x time x (Principal, Interest, Balance)
    # Tranche order: 'CG', 'VE', 'CM', 'GZ', 'TC', 'CZ', 'CA', 'CY'
    # Tranche order:    0,    1,    2,    3,    4,    5,    6,    7
    Tranche_CF_arr = np.zeros((8,241,3))
    for i in range(8):
        Tranche_CF_arr[i,0,2] = Tranche_bal_arr[i]


    #a=time.time()
    Pool1_data = np.zeros((241, 5))
    Pool2_data = np.zeros((241, 5))

    Pool1_data[0, 4] = Pool1_bal
    Pool2_data[0, 4] = Pool2_bal

    lib.pool_cf(Pool1_data, Pool1_mwac, Pool1_age, Pool1_term, SMM_array)
    lib.pool_cf(Pool2_data, Pool2_mwac, Pool2_age, Pool2_term, SMM_array)

    # Reproduce Principal CF Allocation (waterfall)
    Total_principal = Pool1_data[:,2] + Pool2_data[
            :,2] + Pool1_data[:,3] + Pool2_data[:,3]
    CA_CY_princ = 0.225181598 * Total_principal
    Rest_princ = 0.7748184 * Total_principal

    # Temporary arrays for calculating GZ/CZ interest accrual. The "interest" here
    # is not cash flow. 2 columns:
    #interest
    #accrued
    GZ_interest = np.zeros((241, 2))
    CZ_interest = np.zeros((241, 2))

    # Calculate Cash flows
    # redefine giant Tranche_DF
    lib.tranche_CF_calc(Tranche_CF_arr, CA_CY_princ, Rest_princ, GZ_interest,
                        CZ_interest, coupon_rate)

    CF_arr = np.zeros((241, 8))
    for i in range(8):
        CF_arr[:,i] = Tranche_CF_arr[i,:,0] + Tranche_CF_arr[i,:,1]

    return CF_arr[:,-2]

def calc_OAS(zero_df, m, theta_df, kappa, sigma, gamma, p, beta, r0, bond_list, Tranche_bal_arr, wac, tenor, antithetic,
             Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term, Pool2_mwac, Pool2_age, Pool2_term, coupon_rate):
    spot_simulate_df = lib.simulate_rate(m, theta_df, kappa, sigma, r0, antithetic)

    # Calculate 10 yr rates
    tenor_rate = calc_tenor_rate(spot_simulate_df, kappa, sigma, theta_df, tenor)

    v1 = wac-tenor_rate
    v2 = v1.copy()*0.0
    v2[(v2.index.month>=5)&(v2.index.month<=8)] = 1

    smm_df = calc_hazard(gamma, p, beta, v1, v2)

    cash_ser = np.vectorize(calc_cashflow_CA, signature='(n),(n),(),(),(),(),(),(),(),(),(),(k)->(m)')(
                            smm_df.T.values.astype(float), spot_simulate_df.T.values.astype(float),
                            Pool1_bal, Pool2_bal, Pool1_mwac, Pool1_age, Pool1_term,
                            Pool2_mwac, Pool2_age, Pool2_term, coupon_rate,Tranche_bal_arr)

    cash_ser = pd.DataFrame(cash_ser).mean()

    _zero_df = zero_df.copy()
    _zero_df.index = _zero_df['DATE']
    _zero_df = _zero_df[['ZERO']]
    _zero_df = _zero_df.resample('1MS').interpolate(method='index')
    _zero_df['DATE'] = _zero_df.index
    r0 = 0
    oas = scipy.optimize.fsolve(calc_PV_diff, r0, args=(cash_ser, _zero_df, Tranche_bal_arr[-2]))[0]
    return oas

def calc_PV_diff(r, cf, zero_df, par):
    _zero_df = zero_df.copy()
    _zero_df['ZERO'] = _zero_df['ZERO']+r
    discount_df = discount_fac(_zero_df)
    discount_df.index = range(0, len(discount_df))
    pv = (discount_df.iloc[:, 0]*cf).sum()
    # print(pv-par)
    return pv-par

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