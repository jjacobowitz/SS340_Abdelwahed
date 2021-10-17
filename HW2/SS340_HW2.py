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

# wipe the summary file
with open("summary.txt", "w") as f:
    f.truncate()

# =============================================================================
# Useful Functions
# =============================================================================
def run_regression(x, y, xlabel, ylabel, title, save_title):
    # add the x data to the model
    x_sm = sm.add_constant(x)
    
    # fit the model using robust regression
    model = sm.OLS(y, x_sm).fit(cov_type="HC1")
    print(model.summary())
    
    # create and print the stata-style summary table; save to a txt
    stata_summary = summary_col(model, stars=True, float_format='%0.2f')
    print(stata_summary)
    with open('summary.txt', 'a') as f:
        f.write(stata_summary.as_text())
    
    # add new x data for the plot fit line
    x_cont = np.linspace(x.min(), x.max(), 1000)
    x_cont_sm = sm.add_constant(x_cont)
    
    # generate the regression y data
    y_fit = model.predict(x_cont_sm)
    
    # plot the data with the regression line
    plt.figure()
    plt.scatter(x, y, marker='.', label="data")
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
summary_cols = ["fincbtxm",
                "totexpcq",
                "totexppq",
                "alcbevcq",
                "alcbevpq",
                "foodcq",
                "foodpq",
                "fdhomecq",
                "fdhomepq"]
    
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
# save and print the summary statistics of the data; save to a csv
data_19_describe = data_19[summary_cols].describe().round(2)
important_statistics = ["mean", "std", "min", "max"]
data_19_describe.loc[important_statistics].to_csv("SS340_HW2_Describe.csv")
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

# remove individuals that: (1) have 0 food spending, (2) negative expenditures
before = data_19.shape[0]
data_19.drop(data_19[data_19["food"] <= 0].index, inplace=True)
after = data_19.shape[0]
dropped = before - after
print(f"Dropped {dropped} row(s) for food <= 0")

before = data_19.shape[0]
data_19.drop(data_19[data_19["totexp"] <= 0].index, inplace=True)
after = data_19.shape[0]
dropped = before - after
print(f"Dropped {dropped} row(s) for totexp <= 0")

# Summary statistics after cleaning the data; print to console & save to a csv
summary_cols2 = ["fincbtxm",
                 "totexp",
                 "food",
                 "alcbev",
                 "fdhome",
                 "hiincome"]
data_19_describe2 = data_19[summary_cols2].describe().round(2)
data_19_describe2.loc[important_statistics].to_csv("SS340_HW2_Describe2.csv")
print(data_19_describe2)

# =============================================================================
# Linear Regression on All the Data, Part 2(e)
# =============================================================================
# significance for null hypothesis testing
alpha = 0.05

print("\n\nPart 2e")
# linear regression: all
result = run_regression(data_19["fincbtxm"].to_numpy(),
                        data_19["totexp"].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure [$/yr.]",
                        "Expenditure vs Income for All Data",
                        "SS340_HW1_expvsincall.png")

[beta_0, beta_1], p_val, stata_summary = result

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Linear Regression on the High- and Low-Income Data, Part 2(f)
# =============================================================================
print("\n\nPart 2f high-income")
# linear regression: high-income
result = run_regression(data_19["fincbtxm"][data_19["hiincome"]].to_numpy(),
                        data_19["totexp"][data_19["hiincome"]].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure [$/yr.]",
                        "Expenditure vs Income for High-Income HHs",
                        "SS340_HW1_expvsinchiincome.png")

[beta_0, beta_1], p_val, stata_summary = result

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

print("\n\nPart 2f low-income")
# linear regression: low-income
result = run_regression(data_19["fincbtxm"][~data_19["hiincome"]].to_numpy(),
                        data_19["totexp"][~data_19["hiincome"]].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure [$/yr.]",
                        "Expenditure vs Income for Low-Income HHs",
                        "SS340_HW1_expvsincloincome.png")

[beta_0, beta_1], p_val, stata_summary = result

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Linear Regression on Essential and Non-Essential Foods, Part 2(g)
# =============================================================================
print("\n\nPart 2g Essential vs Non-Essential Foods")

print("\nEssential (fdhome)")
result = run_regression(data_19["fincbtxm"].to_numpy(),
                        data_19["fdhome"].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure on Food at Home [$/yr.]",
                        "Expenditure on Food at Home vs Income",
                        "SS340_HW1_fdhomevsincome.png")

[beta_0, beta_1], p_val, stata_summary = result

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("fdhome beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

print("\nNon-Essential (alcbev)")
result = run_regression(data_19["fincbtxm"].to_numpy(),
                        data_19["alcbev"].to_numpy(),
                        "Income [$/yr.]",
                        "Total Expenditure on Alcohol [$/yr.]",
                        "Expenditure on Alcohol vs Income",
                        "SS340_HW1_alcbevvsincome.png")

[beta_0, beta_1], p_val, stata_summary = result

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("alcbev beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)