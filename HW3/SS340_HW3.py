# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
SS340 Cause and Effect
Homework 3
Created: 10/23/2021
Due: 11/01/2021

Python program for Homework 3
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from scipy.stats import t, ttest_ind
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
        f.write(title)
        f.write(stata_summary.as_text())
        f.write("\n\n\n")
    
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

def confidence(alpha, single_tailed=True):
    """calculates the single- or double-tailed confidence"""
    if single_tailed:
        return 1-alpha
    else:
        return 1-(alpha/2)
    
def t_tabulated(conf, dof):
    """calculates the tabulated t-statistic"""
    return t.ppf(conf, dof)

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
# Read data and clean it
# =============================================================================
cols = ["dmage", "mrace3", "dmar", "dlivord", "frace4", "dgestat", "csex", 
        "dbrwt", "dplural", "cigar", "drink", "tobacco", "wgain", "lung", 
        "cardiac"]

data = pd.read_csv("pennbirthwgt0.csv", usecols=cols)

# new column names
rename = ["agemother", "whitemother", "married", "livebirths", 
          "whitefather", "gestationweek", "male", "birthweight", "kidsatonce", 
          "cigsperday", "etohperweek", "tobaccouse", "pregweightgain", 
          "lungdisease", "cardiacdisease"]

data.rename(columns={old:new for old, new in zip(cols, rename)}, inplace=True)

# the values for missing data for each variable
missing = {"livebirths":99,
           "whitefather":4,
           "gestationweek":99,
           "birthweight":9999,
           "cigsperday":99,
           "etohperweek":99,
           "tobaccouse":9,
           "pregweightgain":99,
           "lungdisease":[8, 9],
           "cardiacdisease":[8, 9]}

# remove rows with missing data
for k, v in missing.items():
    if isinstance(v, list):
        for value in v:
            data.drop(data[data[k] == value].index, inplace=True)
    else:
        data.drop(data[data[k] == v].index, inplace=True)

# convert boolean columns to 1 or 0 by replacing not 1s with 0s
booleans = ["whitemother", "married", "whitefather", "male", "tobaccouse", 
            "lungdisease", "cardiacdisease"]
for boolean in booleans:
    data[boolean].replace([2, 3], 0, inplace=True)
    
data["smoker"] = data["cigsperday"] > 0

# =============================================================================
# Summary Statistics (Part 1e)
# =============================================================================
important_statistics = ["mean", "std", "min", "max"]
data_describe = data.describe().loc[important_statistics].round(2)
data_describe.loc[important_statistics].to_csv("SS340_HW3_Describe.csv")
print(data_describe)    

# =============================================================================
# Mean Difference in Birthweight by Smoking (Parts 2a and 2b)
# =============================================================================
mean_birthweight_notsmoker = data["birthweight"][~data["smoker"]].mean()
mean_birthweight_smoker = data["birthweight"][data["smoker"]].mean()

print(f"Mean birthweight not-smoker: {mean_birthweight_notsmoker:.2f}g")
print(f"Mean birthweight smoker: {mean_birthweight_smoker:.2f}g")

# hypothesis: average birthweight is similar for smokers and non-smokers
# H0: smoker_mean_birthweight - nonsmoker_mean_birthweight = 0
# Ha: smoker_mean_birthweight - nonsmoker_mean_birthweight != 0   
alpha = 0.05
conf = confidence(alpha, single_tailed=False)
n = len(data)
dof = n - 2     # -2 because there are two means used

# mean-mean null hypothesis
t_stat, p_val = ttest_ind(data["birthweight"][~data["smoker"]], 
                          data["birthweight"][data["smoker"]], 
                          equal_var=False)

print("birthweight-smoker null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)
t_tab = t_tabulated(conf, dof)
print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")

# =============================================================================
# Simple Regression (Part 2e)
# =============================================================================
print("\nSimple Regression")
result = run_regression(data["cigsperday"].to_numpy(),
                        data["birthweight"].to_numpy(),
                        "Cigarettes Smoked [cigs./day]",
                        "Child Birthweight [g]",
                        "Birthweight vs. Cigarettes Smoked Per Day",
                        "SS340_HW3_simpleregress.png")

[beta_0, beta_1], p_val, stata_summary = result

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("birthweight-cigs beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)
    