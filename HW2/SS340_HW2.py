# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
SS340 Cause and Effect
Homework 2
Created: 10/06/2021
Due: 10/18/2021

Python program for Homework 2
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt

# close any currently open plots
plt.close("all")

# =============================================================================
# Useful Functions
# =============================================================================
def run_regression(x, y, xlabel, ylabel, title, save_title):
    x_sm = sm.add_constant(x)
    model = sm.OLS(y, x_sm).fit()
    print(model.summary())
    
    stata_summary = summary_col(model, stars=True, float_format='%0.3f')
    print(stata_summary)
    
    x_cont = np.linspace(x.min(), x.max(), 1000)
    x_cont_sm = sm.add_constant(x_cont)
    y_fit = model.predict(x_cont_sm)
    
    plt.figure()
    plt.scatter(x, y, label="data")
    plt.plot(x_cont, y_fit, 'r', label="fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(save_title)
    
    return model.params, model.pvalues[1], stata_summary

def reject_null(p_val, alpha):
    """returns True if the null hypothesis is rejected, False if it fails to be
    rejected. Bases the test on the p-value.
    """
    return p_val <= alpha

def null_hypothesis_test(p_val, alpha):
    """runs the null hypothesis test"""
    if reject_null(p_val, alpha):
        print(f"Reject the null hypothesis with alpha = {alpha*100}%")
    else:
        print(f"Fail to reject the null hypothesis with alpha = {alpha*100}%")

# =============================================================================
# Data Import, Part 2(a)
# =============================================================================
# I put the important column names into a txt file to read in. 
# It is much cleaner.
with open("important_columns.txt", "r") as f:
    columns = f.read().split(",")


# columns that I want to summarize
summary_cols = set(columns) - {"newid", "fam_size", "num_auto"}
    
# import the 4 quarters of data
data_19q1 = pd.read_csv("fmli191x.csv", usecols=columns)
data_19q2 = pd.read_csv("fmli192.csv", usecols=columns)
data_19q3 = pd.read_csv("fmli193.csv", usecols=columns)
data_19q4 = pd.read_csv("fmli194.csv", usecols=columns)

# combine into one variable
data_19 = pd.concat([data_19q1, data_19q2, data_19q3, data_19q4], 
                    ignore_index=True)

# =============================================================================
# Summary Statistics, Part 2(b)
# =============================================================================
# save and print the summary statistics of the data
data_19_describe = data_19[summary_cols].describe().round(2)
print(data_19_describe)

# =============================================================================
# Clean the Data, Part 2(c)
# =============================================================================
# high-income dummy variable. high-income is defined as income > median income
data_19["hiincome"] = data_19["fincbtxm"] > data_19["fincbtxm"].median()

# expenditure variables combined for CQ and PQ
data_19["totexp"] = data_19[["totexpcq", "totexppq"]].sum(axis=1)*4
data_19["food"] = data_19[["foodcq", "foodpq"]].sum(axis=1)*4
data_19["alcbev"] = data_19[["alcbevcq", "alcbevpq"]].sum(axis=1)*4
data_19["fdhome"] = data_19[["fdhomecq", "fdhomepq"]].sum(axis=1)*4
data_19["fdmap"] = data_19[["fdmapcq", "fdmappq"]].sum(axis=1)*4
data_19["fdaway"] = data_19[["fdawaycq", "fdawaypq"]].sum(axis=1)*4
data_19["majapp"] = data_19[["majappcq", "majapppq"]].sum(axis=1)*4
data_19["tentrmn"] = data_19[["tentrmnc", "tentrmnp"]].sum(axis=1)*4
data_19["educa"] = data_19[["educacq", "educapq"]].sum(axis=1)*4
data_19["elctrc"] = data_19[["elctrccq", "elctrcpq"]].sum(axis=1)*4

# remove individuals that: (1) have 0 food spending, (2) negative expenditures
before = data_19.shape[0]
data_19.drop(data_19[data_19["food"] <= 0].index, inplace=True)
after = data_19.shape[0]
dropped = before - after
print(f"Dropped {dropped} row(s) for food <= 0")

# rows that contain expenditures (besides food)
exp_cols = ["totexp",
            "alcbev",
            "fdhome",
            "fdmap",
            "fdaway",
            "majapp",
            "tentrmn",
            "educa",
            "elctrc"]
before = data_19.shape[0]
for col in exp_cols:
    data_19.drop(data_19[data_19[col] < 0].index, inplace=True)
after = data_19.shape[0]
dropped = before - after
print(f"Dropped {dropped} row(s) for expenditures < 0")

# Summary statistics after cleaning the data
summary_cols2 = set(data_19.columns) - {"newid", "fam_size", "num_auto"}
data_19_describe2 = data_19[summary_cols2].describe().round(2)
print(data_19_describe2)

# =============================================================================
# Linear Regression on All the Data, Part 2(e)
# =============================================================================
# significance for null hypothesis testing
alpha = 0.05

# linear regression: all
result = run_regression(data_19["fincbtxm"].to_numpy(),
                        data_19["totexp"].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure [$/qtr.]",
                        "Expenditure vs Income for All Data",
                        "SS340_HW1_expvsincall.png")

[beta_0, beta_1], p_val, stata_summary = result

print("\nPart 2e")
print(f"{beta_0=:.3f}, {beta_1=:.3f}")
print("beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Linear Regression on the High- and Low-Income Data, Part 2(f)
# =============================================================================
# linear regression: high-income
result = run_regression(data_19["fincbtxm"][data_19["hiincome"]].to_numpy(),
                        data_19["totexp"][data_19["hiincome"]].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure [$/qtr.]",
                        "Expenditure vs Income for High-Income HHs",
                        "SS340_HW1_expvsinchiincome.png")

[beta_0, beta_1], p_val, stata_summary = result

print("\nPart 2f high-income")
print(f"{beta_0=:.3f}, {beta_1=:.3f}")
print("beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# linear regression: low-income
result = run_regression(data_19["fincbtxm"][~data_19["hiincome"]].to_numpy(),
                        data_19["totexp"][~data_19["hiincome"]].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure [$/qtr.]",
                        "Expenditure vs Income for Low-Income HHs",
                        "SS340_HW1_expvsincloincome.png")

[beta_0, beta_1], p_val, stata_summary = result

print("\nPart 2f low-income")
print(f"{beta_0=:.3f}, {beta_1=:.3f}")
print("beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Linear Regression on Essential and Non-Essential Foods, Part 2(g)
# =============================================================================
print("\nPart 2g Essential vs Non-Essential Foods")

result = run_regression(data_19["fincbtxm"].to_numpy(),
                        data_19["fdhome"].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure on Food at Home [$/qtr.]",
                        "Expenditure on Food at Home vs Income",
                        "SS340_HW1_fdhomevsincome.png")

[beta_0, beta_1], p_val, stata_summary = result

print("Essential")
print(f"{beta_0=:.3f}, {beta_1=:.3f}")
print("beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

result = run_regression(data_19["fincbtxm"].to_numpy(),
                        data_19["alcbev"].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure on Alcohol [$/qtr.]",
                        "Expenditure on Alcohol vs Income",
                        "SS340_HW1_alcbevvsincome.png")

[beta_0, beta_1], p_val, stata_summary = result

print("Non-Essential")
print(f"{beta_0=:.3f}, {beta_1=:.3f}")
print("beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)