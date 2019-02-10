#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 17:32:44 2019

@author: paul
"""

import numpy as np


def log_log_grad(param, tb, te, event, covars):
    """
    This function calculates the gradient of the log-likelihood for the
    proportional hazard model using the log-logistics baseline distribution
    """
    tb = tb.flatten()
    te = te.flatten()
    event = event.flatten()
    g = param[0] # Amplitude of the baseline hazard; gamma in the notation
    p = param[1] # Shape of baseline hazard; p in the notation
    coef = param[2:] # Coefficients for covariates; beta in the notation

    dlldg1 = sum(event*(p/g-(p*g**(p-1)*(te**p))/(1+(g*te)**p)))
    if len(covars):
        dlldg2 = sum((p*g**(p-1))*((te**p/(1+(g*te)**p))-(tb**p/(1+(g*tb)**p)))*np.exp(covars.dot(coef)))
    else:
        dlldg2 = sum((p*g**(p-1))*((te**p/(1+(g*te)**p))-(tb**p/(1+(g*tb)**p))))

    dlldg = -(dlldg1-dlldg2)

    dlldp1 = sum(event*(1/p+np.log(g*te)-(g*te)**p*np.log(g*te)/(1+(g*te)**p)))

    # When tb = 0, calculate the derivative of the unconditional survival function.
    # This is because the derivative of the conditional survival function does not
    # generalize to the unconditional case when tb = 0. There is a singularity on
    # log(g*tb) for tb = 0.

    ln_gtb = np.log(g*tb)
    ln_gtb[np.isposinf(ln_gtb)] = 0
    ln_gtb[np.isneginf(ln_gtb)] = 0

    if len(covars):
        dlldp2 = sum((((g*te)**p)*np.log(g*te)/(1+(g*te)**p)-(g*te)**p*ln_gtb/(1+(g*tb)**p))*np.exp(covars.dot(coef)))
    else:
        dlldp2 = sum((((g*te)**p)*np.log(g*te)/(1+(g*te)**p)-(g*te)**p*ln_gtb/(1+(g*tb)**p)))

    dlldp = -(dlldp1-dlldp2)

    grad = [dlldg, dlldp]

    for i in range(0, len(coef)):
        dlldc1 = sum(event*covars[:,i])
        dlldc2 = sum((np.log(1+(g*te)**p)-np.log(1+(g*tb)**p))*np.exp(covars.dot(coef))*covars[:,i])
        dlldc = -(dlldc1-dlldc2)

        grad.append(dlldc)
    return grad


def log_log_like(param, tb, te, event, covars):
    """
    This function calculates the log likelihood for a proportional hazard
    model with log-logistic baseline hazard.  It can be used to solve for
    the parameters of the model.
    """
    # tb=static_df['period_begin']/365;te=static_df['period_end']/365
    # event=static_df['prepay'];param=[0.1]*7
    tb = tb.flatten()
    te = te.flatten()
    event = event.flatten()
    # Get the number of parameters
    nentries = len(te)

    g = param[0]  # Amplitude of the baseline hazard; gamma in the notation
    p = param[1]  # Shape of baseline hazard; p in the notation
    coef = param[2:]  # Coefficients for covariates; beta in the notation

    # The following variables are vectors with a row for each episode
    # Log of baseline hazard
    logh = (np.log(p) + np.log(g) + (p-1)*(np.log(g)+np.log(te)) - np.log(1+(g*te)**p))

    logc = np.zeros(nentries)
    logF = -(np.log(1+(g*te)**p) - np.log(1+(g*tb)**p))
    if not len(covars):
        # Product of covarites and coefficients
        logc = (covars.dot(coef)).flatten()
        # Log of conditional survival function
        logF = logF*np.exp(covars.dot(coef))

    # Construct the negative of log likelihood
    neglogL = -(sum(event*(logh+logc)) + sum(logF))

    # Calculate the derivative of the log likelihood with respect to each parameter.
    # In order for the maximum likelihood estimation to converge it is necessary to
    # provide these derivatives so that the search algogrithm knows which direction
    # to search in.

    grad = log_log_grad(param, tb, te, event, covars)
    return neglogL, grad


def hessian(x, x_grad):
    """
    Calculate the hessian matrix with finite differences

    Args:
       x : ndarray

       x_grad : ndarray

    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    # x=sol.reshape(-1, 1); x_grad=np.array(sol_grad).reshape(-1, 1)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian