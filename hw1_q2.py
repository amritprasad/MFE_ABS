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
Pool1_bal=77657656.75
Pool2_bal=128842343.35

# Coupon rates
Pool1_wac=5.402
Pool2_wac=5.419
Pool1_mrate=Pool1_wac/12
Pool2_mrate=Pool2_wac/12

# CPR - constant (not dependent on interest rate)
CPR = 150

#%%
# make function for this

# initialize numpy array