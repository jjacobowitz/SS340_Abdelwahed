# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:14:57 2021

@author: Jared
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

x = np.linspace(0, 100, 1000)
slope = np.random.rand(len(x))
intercept = np.random.rand(len(x))
y = slope*x + intercept

# add the x data to the model
x_sm = sm.add_constant(x)

# fit the model using robust regression
model = sm.OLS(y, x_sm).fit(cov_type="HC1")

# print the normal summary
print(model.summary())

# rint the stata-style summary table
print(summary_col(model, stars=True))

