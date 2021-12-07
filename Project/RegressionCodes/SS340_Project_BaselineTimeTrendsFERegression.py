# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Baseline regression with time trends and time fixed effects for the project

Includes county FE, linear time trends, and time fixed effects

Regression:
    disasters_{i,t} = b0 + b1 tempc_{i,t} + b2 t + a_i + T_t + e_{i,t}
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
y = y.to_numpy().reshape(-1, 1)
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

with open('../Results/baseline_regression_with_both_summary.txt',
          'w') as f:
    f.write("Baseline Regression with Both")
    f.write('\n')
    f.write(stata_summary.as_text())
print("Done.")
