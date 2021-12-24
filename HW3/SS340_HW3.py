# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
SS340 Cause and Effect
Homework 3
Created: 10/23/2021
Due: 11/01/2021

Python program for Homework 3

Note: The order of operations do not match the HW, so Part 1d is done before
part 1a
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from scipy.stats import t, ttest_ind
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# close any currently open plots
plt.close("all")

# wipe the summary files
with open("results/normal_summary.txt", "w") as f:
    f.truncate()

# wipe the summary file
with open("results/stata_summary.txt", "w") as f:
    f.truncate()

# =============================================================================
# Useful Functions
# =============================================================================


def run_regression(x, y,
                   xlabel=None, ylabel=None, title=None, save_title=None):
    # add the x data to the model
    X = sm.add_constant(x)

    # fit the model using robust regression
    model = sm.OLS(y, X).fit(cov_type="HC1")

    if None not in [xlabel, ylabel, title, save_title]:
        # add new x data for the plot fit line
        x_cont = np.linspace(x.min(), x.max(), 1000)
        X = sm.add_constant(x_cont)

        # generate the regression y data
        y_fit = model.predict(X)

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

    return model, model.params, model.pvalues[1]


def get_normal_model_summary(model, title):
    # create the normal-style summary table and save to a txt
    normal_summary = model.summary()
    with open('results/normal_summary.txt', 'a') as f:
        f.write(title)
        f.write('\n')
        f.write(normal_summary.as_text())
        f.write("\n\n\n")

    return normal_summary


def get_stata_model_summary(model, name, title):
    # create the stata-style summary table and save to a txt
    stata_summary = summary_col(
        model, model_names=name, stars=True, float_format="%0.2f")
    with open('results/stata_summary.txt', 'a') as f:
        f.write(title)
        f.write('\n')
        f.write(stata_summary.as_text())
        f.write("\n\n\n")

    return stata_summary


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
# Read in data
# =============================================================================
cols = ["dmage", "mrace3", "dmar", "dlivord", "frace4", "dgestat", "csex",
        "dbrwt", "dplural", "cigar", "drink", "wgain", "lung",
        "cardiac"]

data = pd.read_csv("pennbirthwgt0.csv", usecols=cols)

# =============================================================================
# Rename columns (Part 1d)
# =============================================================================
# new column names
rename = ["agemother", "whitemother", "married", "livebirths",
          "whitefather", "gestationweek", "male", "birthweight", "kidsatonce",
          "cigsperday", "etohperweek", "pregweightgain", "lungdisease",
          "cardiacdisease"]

data.rename(columns={old: new for old, new in zip(cols, rename)}, inplace=True)

# =============================================================================
# Remove missing (Parts 1a and 1c)
# =============================================================================
# the values for missing data for each variable
missing = {"livebirths": 99,
           "whitefather": 4,
           "gestationweek": 99,
           "birthweight": 9999,
           "cigsperday": 99,
           "etohperweek": 99,
           "pregweightgain": 99,
           "lungdisease": [8, 9],
           "cardiacdisease": [8, 9]}

# remove rows with missing data
for k, v in missing.items():
    if isinstance(v, list):
        for value in v:
            data.drop(data[data[k] == value].index, inplace=True)
    else:
        data.drop(data[data[k] == v].index, inplace=True)

# =============================================================================
# Correct dummy variables (Part 1b)
# =============================================================================
# convert boolean columns to 1 or 0 by replacing not 1s with 0s
booleans = ["whitemother", "married", "whitefather", "male", "lungdisease",
            "cardiacdisease"]
for boolean in booleans:
    data[boolean].replace([2, 3], 0, inplace=True)

data["smoker"] = data["cigsperday"] > 0
smoker = data["smoker"]         # easier reference to the smoker dummy variable

# =============================================================================
# Summary Statistics (Part 1e)
# =============================================================================
important_statistics = ["mean", "std", "min", "max"]
data_describe = data.describe().loc[important_statistics].transpose().round(2)
data_describe[important_statistics].to_csv("results/SS340_HW3_Describe.csv")
print(data_describe)

# =============================================================================
# Mean Difference in Birthweight by Smoking (Parts 2a and 2b)
# =============================================================================
mean_birthweight_nonsmoker = data["birthweight"][~smoker].mean()
mean_birthweight_smoker = data["birthweight"][smoker].mean()

print("\n")
print(f"Mean birthweight not-smoker: {mean_birthweight_nonsmoker:.2f}g")
print(f"Mean birthweight smoker: {mean_birthweight_smoker:.2f}g")

# hypothesis: average birthweight is similar for smokers and non-smokers
# H0: smoker_mean_birthweight - nonsmoker_mean_birthweight = 0
# Ha: smoker_mean_birthweight - nonsmoker_mean_birthweight != 0
alpha = 0.05
conf = confidence(alpha, single_tailed=False)
n = len(data)
dof = n - 2     # -2 because there are two means used

# mean-mean null hypothesis
t_stat, p_val = ttest_ind(data["birthweight"][~smoker],
                          data["birthweight"][smoker],
                          equal_var=False)

print("\n")
print("birthweight-smoker null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)
t_tab = t_tabulated(conf, dof)
print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")

# =============================================================================
# Table of differences (Part 2c)
# =============================================================================
# variables we think might show the populations of mothers are not comparable
of_interest = ["birthweight", "agemother", "whitemother", "married",
               "livebirths", "gestationweek", "kidsatonce", "etohperweek",
               "pregweightgain", "cardiacdisease"]

# skeleton of the table with the desired columns
table_of_diffs = pd.DataFrame(columns=["Mean",
                                       "Smoker",
                                       "Not Smoker",
                                       "Difference",
                                       "p-value"])

table_of_diffs.index.name = "Variable"  # rename the index column to variable

# Creates a table showing the values for each group, their differences, and
# the p-value, where the null hypothesis is that they are equal. If the null
# hypothesis is rejected (p<0.05) then there is a difference, and the groups
# are not equal
for interest in of_interest:
    mean = data[interest].mean()
    smoker_data = data[interest][smoker]
    nonsmoker_data = data[interest][~smoker]
    smoker_result = smoker_data.mean()
    nonsmoker_result = nonsmoker_data.mean()
    diff = smoker_result - nonsmoker_result
    _, p_value = ttest_ind(smoker_data, nonsmoker_data, equal_var=False)
    row = {"Mean": mean,
           "Smoker": smoker_result,
           "Not Smoker": nonsmoker_result,
           "Difference": diff,
           "p-value": round(p_value, 2)}  # round to 2 decimals for clarity
    row = pd.Series(row, name=interest)
    table_of_diffs = table_of_diffs.append(row)
table_of_diffs.to_csv("results/SS340_HW3_TableofDiffs.csv")

# =============================================================================
# Correlation matrix (Part 2d)
# =============================================================================
# DataFrame.corr() creates a correlation matrix of the DataFrame
corr_matrix = data[of_interest].corr()
print("\n")
print(corr_matrix.round(2))

# OVB is a table where the birthweight and cigsperday columns are the
# correlation values with the variable and the sign column is the sign of the
# OVB based on the correlations
# probably a better way to do this..

# skeleton of the table with the desired columns
OVB = pd.DataFrame(columns=["birthweight", "cigsperday", "sign"])

# fill the table with the values
for interest in of_interest[1:]:
    sign = 1    # will be the sign of the OVB

    # add the rows without data
    row = pd.Series({"birthweight": np.nan,
                     "cigsperday": np.nan,
                     "sign": np.nan},
                    name=interest)
    OVB = OVB.append(row)

    # calculate the signs
    for col in OVB.columns:
        if col != "sign":
            corr = data[[col, interest]].corr().round(2).to_numpy()[0, 1]
            OVB[col][interest] = corr
            sign *= np.sign(corr)
        else:
            OVB[col][interest] = sign
OVB.to_csv("results/SS340_HW3_OVB.csv")

# =============================================================================
# Simple Regression (Part 2e)
# =============================================================================
print("\nSimple Regression")
title = "Birthweight vs. Cigarettes Smoked Per Day"
xlabel = "Cigarettes Smoked [cigs./day]"
ylabel = "Child Birthweight [g]"
result = run_regression(data["cigsperday"],
                        data["birthweight"],
                        xlabel=xlabel,
                        ylabel=ylabel,
                        title=title,
                        save_title="figures/SS340_HW3_simpleregress.png")

model, [beta_0, beta_1], p_val = result
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "birthweight", title)


print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("birthweight-cigs beta_0 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Adding more covariates (Part 2f)
# =============================================================================
covariates = ["cigsperday",
              "married",
              "livebirths",
              "gestationweek",
              "whitemother",            # until here: good controls
              "agemother",
              "whitefather",
              "male",
              "kidsatonce",             # until here: useless controls
              "pregweightgain", ]       # until here: bad controls
models = []
names = []
for indx, covariate in enumerate(covariates):
    xdata = data[covariates[:indx+1]]
    ydata = data["birthweight"]
    model, _, _ = run_regression(xdata, ydata)
    models.append(model)
    names.append(f"({indx})")

stata_summary = get_stata_model_summary(models, names, "Testing Covariates")
