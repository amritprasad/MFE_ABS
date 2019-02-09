# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:02:38 2019
This file reads in the data for the REMIC from the excel file.

We get the basic parameters (principle, coupon, CPR) and calculate 
    the expected cash flows.

@author: willd
"""

import numpy as np
import pandas as pd

#%% 
# We hardcode the parameters below, but we can just as easily pull it from
# the sheet directly

#pool_data = pd.read_excel('REMIC_Template New.xlsx', 
#                          sheetname='Pool Info', nrows=3)

#%% Hard coded parameters

# Starting Balance 
Pool1_bal   = 77657656.75
Pool2_bal   = 128842343.35
Pool1_term  = 236
Pool2_term  = 237

# Coupon rates
Pool1_wac   = 0.05402
Pool2_wac   = 0.05419
Pool1_mwac = Pool1_wac/12
Pool2_mwac = Pool2_wac/12
Pool1_age   = 3
Pool2_age   = 3
coupon_rate = .05/12

# PSA - constant (not dependent on interest rate)
PSA = 1.50

#%% Calculate Pool principal payments. The principal payments pass through
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
    if abs(bal - 0)<1e-2:
        return 0.0
    
    return bal * rate * (1+rate)**n_per / ( (1+rate)**n_per - 1)

# Keep separate CPR array so that we can make it depend on the short rate path
def cpr(PSA, pp_rate, total_term, age, rate_path=None):
    """
    Function to give us CPR schedule
    """
    arr       = np.linspace(0,total_term-1,num=total_term,dtype=int)
    CPR_array = PSA * pp_rate * np.minimum(1,(arr+age)/30)
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
#%% Produce Pool cash flows
CPR_array=cpr(PSA,.06,241,3)
SMM_array = 1 - (1-CPR_array)**(1/12)

# pre-allocate arrays to hold pool cash flows
# 7 columns: PMT, Interest, Principal, Pre-pmt CPR, SMM, pp CF, Balance
cols       = ['PMT', 'Interest', 'Principal', 'PP CF', 'Balance']
Pool1_data = pd.DataFrame( np.zeros((241,5)), columns=cols)
Pool2_data = pd.DataFrame( np.zeros((241,5)), columns=cols)

Pool1_data.loc[0,'Balance'] = Pool1_bal
Pool2_data.loc[0,'Balance'] = Pool2_bal
pool_cf(Pool1_data, Pool1_mwac, Pool1_age, Pool1_term, SMM_array)
pool_cf(Pool2_data, Pool2_mwac, Pool2_age, Pool2_term, SMM_array)

#%% Reproduce Principal CF Allocation (waterfall)
Total_principal = Pool1_data['Principal'] + Pool2_data['Principal'] + Pool1_data['PP CF'] \
                    + Pool2_data['PP CF']
CA_CY_princ = 0.225181598 * Total_principal
Rest_princ = 0.7748184 * Total_principal

# Pre-Allocate for each Tranche, then do waterfall logic
Tranche_dict = {}
Tranche_list = ['CG', 'VE', 'CM',	'GZ', 'TC',	'CZ', 'CA', 'CY']
cols=['Principal','Interest','Balance']

Tranche_bal_dict={
'CG':74800000,
'VE':5200000,
'CM':14000000,
'GZ':22000000,
'TC':20000000,
'CZ':24000000,
'CA':32550000,
'CY':13950000
}

# small for loop, pre-allocate data for tranches
for i in Tranche_list:
    Tranche_dict[i] = pd.DataFrame(np.zeros((241,3)),columns=cols)
    Tranche_dict[i].loc[0,'Balance'] = Tranche_bal_dict[i]

# temporary arrays for calculating GZ/CZ interest and accrual
cols=['interest','accrued']
GZ_interest = pd.DataFrame( np.zeros((241,2)), columns=cols)
CZ_interest = pd.DataFrame( np.zeros((241,2)), columns=cols)

#%%            
# Reproduce cashflow waterfall for the two groups
def tranche_CF_calc(Tranche_dict,CA_CY_princ,Rest_princ,GZ_interest,CZ_interest):
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
    
#%% Output Total cash flows for each Tranche
tranche_CF_calc(Tranche_dict,CA_CY_princ,Rest_princ,GZ_interest,CZ_interest)
                                                
CF_df = pd.DataFrame( np.zeros((240,8)), columns=list(Tranche_bal_dict.keys()) )

for i in Tranche_bal_dict.keys():
    CF_df[i] = Tranche_dict[i]['Principal'] + Tranche_dict[i]['Interest']