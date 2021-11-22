# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Baseline regression for the project
"""
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

print("Loading data...", end="")
data = pd.read_csv("../Datasets/CombinedData.csv",
                    usecols=("fips", "year", "tempc", "disasters"))
print("Done.")

print("Creating endogenous variable...", end="")
y = data.pop("disasters")
y = y.to_numpy().reshape(-1,1)
print("Done.")

print("Creating Dummies...", end="")
X = pd.get_dummies(data, columns=["year", "fips"], drop_first=True)
print("Done.")

print("Running Regression...", end="")
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm).fit(cov_type="HC1")
print("Done.")

print("Saving Summary...", end="")
stata_summary = summary_col(model, 
                            stars=True, 
                            float_format="%0.2f",
                            regressor_order=["const", "tempc"],
                            drop_omitted=True)

with open('../Results/baseline_regression_summary.txt', 'w') as f:
    f.write("Baseline Regression")
    f.write('\n')
    f.write(stata_summary.as_text())
print("Done.")