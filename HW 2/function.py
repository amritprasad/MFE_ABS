"""
MFE 230M
Library of functions (HW 2)
"""
# Imports
import pandas as pd
import numpy as np

# Import library
import abslibrary as lib  # HW 1 library

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

def calc_bond_price(cf_bond, _r):
    r = _r.copy()
    r = r.astype(float)
    r = r.iloc[:len(cf_bond)]
    r.index = cf_bond.index

    R = (cf_bond.sum(1).T*(np.exp(r/24)-1).T).T

    r_cum = r.cumsum()

    price_dict = {}
    for i in cf_bond.columns:
        price_dict[i] = (cf_bond[i].T/np.exp(r_cum/12).T).T.sum()
    price_dict['R'] = (R/np.exp(r_cum/12)).sum()

    price_ser = pd.Series(price_dict)
    return price_ser