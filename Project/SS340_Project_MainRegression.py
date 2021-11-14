# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Main regression analysis
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.base.distributed_estimation import DistributedModel
from statsmodels.iolib.summary2 import summary_col

print("Loading data...", end="")
data = pd.read_csv("CombinedData.csv",
                    usecols=("fips", "year", "tempc", "disasters"))
print("Done.")

print("Creating endogenous variable...", end="")
y = data.pop("disasters")
y = y.to_numpy().reshape(-1,1)
print("Done.")

print("Creating Dummies...", end="")
X = pd.get_dummies(data, columns=["year", "fips"], drop_first=True).to_numpy()
print("Done.")

print("Running Regression...", end="")
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm).fit(cov_type="HC1")
print("Done.")

print("Saving Summary...", end="")
stata_summary = summary_col(model, 
                            stars=True, 
                            float_format="%0.2f",
                            regressor_order=["tempc"],
                            drop_omitted=True)

with open('stata_summary.txt', 'w') as f:
    f.write("Main Regression")
    f.write('\n')
    f.write(stata_summary.as_text())
    f.write("\n\n\n")
print("Done.")