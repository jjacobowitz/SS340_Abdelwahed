# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
SS340 Cause and Effect
Homework 3
Created: 11/13/2021
Due: 11/22/2021

Python program for Homework 4
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.iv import IV2SLS as lmIV2SLS
# from scipy.stats import t, ttest_ind

# close any currently open plots
plt.close("all")

# wipe the summary files
with open("normal_summary.txt", "w") as f:
    f.truncate()

with open("stata_summary.txt", "w") as f:
    f.truncate()

# =============================================================================
# Useful Functions
# =============================================================================
def plot_regression(model, x, y, xlabel, ylabel, title, save_title):
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
    
def run_regression(x, y, 
                   xlabel=None, ylabel=None, title=None, save_title=None):
    # add the x data to the model
    X = sm.add_constant(x)
    
    # fit the model using robust regression
    model = sm.OLS(y, X).fit(cov_type="HC1")
    if None not in [xlabel, ylabel, title, save_title]:
        plot_regression(model, x, y, xlabel, ylabel, title, save_title)
    
    return model, model.params, model.pvalues[1]

def get_normal_model_summary(model, title):
    # create the normal-style summary table and save to a txt
    normal_summary = model.summary()
    with open('normal_summary.txt', 'a') as f:
        f.write(title)
        f.write('\n')
        f.write(normal_summary.as_text())
        f.write("\n\n\n")
        
    return normal_summary

def get_stata_model_summary(model, name, title):    
    # create the stata-style summary table and save to a txt
    stata_summary = summary_col(model, 
                                model_names=name, 
                                stars=True, 
                                float_format="%0.2f")
    with open('stata_summary.txt', 'a') as f:
        f.write(title)
        f.write('\n')
        f.write(stata_summary.as_text())
        f.write("\n\n\n")
        
    return stata_summary

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
# Import Data for Part 1
# =============================================================================
df1 = pd.read_stata("Simulated.dta")

important_statistics = ["mean", "std", "min", "max"]
data_describe1 = df1.describe().loc[important_statistics].transpose().round(2)
data_describe1[important_statistics].to_csv("SS340_HW4_Describe1.csv")
print(data_describe1)    

# =============================================================================
# Scatter with Regression Line (Part 1a and 1b)
# =============================================================================
print(" Parts 1a and 1b ".center(100, "="))
result = run_regression(df1.X1, df1.Y1, 
                        xlabel="X1",
                        ylabel="Y1",
                        title="Y1 vs X1 (Parts 1a and 1b)",
                        save_title="SS340_HW4_Parts1a1b.png")
model, [beta_0, beta_1], p_val = result

title = "Y1 and X1 for Parts 1a and 1b"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "Y1", title)

alpha = 0.05
print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("Y1-X1 beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Scatter with Regression Line (Part 1c)
# =============================================================================
print(" Part 1c ".center(100, "="))
result = run_regression(df1.X2, df1.Y2, 
                        xlabel="X2",
                        ylabel="Y2",
                        title="Y2 vs X2 (Part 1c)",
                        save_title="SS340_HW4_Part1c.png")
model, [beta_0, beta_1], p_val = result

title = "Y2 and X2 for Part 1c"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "Y2", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("Y2-X2 beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# 2SLS "By Hand" (Part 1d)
# =============================================================================
print(" Part 1d ".center(100, "="))

# First Stage
result = run_regression(df1.Z, df1.X2, 
                        xlabel="Z",
                        ylabel="X2",
                        title="X2 vs Z (Part 1d(1))",
                        save_title="SS340_HW4_Part1d_1.png")
model, [beta_0, beta_1], p_val = result

title = "X2 and Z for Part 1d(1)"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "X2", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("X2-Z beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# Second Stage
Z = sm.add_constant(df1.Z)
X2_hat = model.predict(Z)

result = run_regression(X2_hat.to_numpy().reshape(-1,1), df1.Y2, 
                        xlabel="$\widehat{X2}$",
                        ylabel="Y2",
                        title="Y2 vs $\widehat{X2}$ (Part 1d(2))",
                        save_title="SS340_HW4_Part1d_2.png")
model, [beta_0, beta_1], p_val = result

title = "Y2 and X2 for Part 1d(2)"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "Y2", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("Y2-X2_hat beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# 2SLS statsmodel (Part 1e)
# =============================================================================
print(" Part 1e ".center(100, "="))
X2 = sm.add_constant(df1.X2)
Z = sm.add_constant(df1.Z)
model = IV2SLS(endog=df1.Y2, exog=X2, instrument=Z)
model.cov_type = "HC1"
model = model.fit()
[beta_0, beta_1] = model.params
p_value = model.pvalues[1]

title = "Y2 and X2 for Part 1e"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "Y2", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("Y2-X2 beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Import Data for Part 2
# =============================================================================
# only importing the useful columns
df2 = pd.read_csv("schooling_earnings.csv", 
                  usecols=("log_earnings", 
                           "yrsed", 
                           "dist", 
                           "dadcoll", 
                           "momcoll"))

# rename columns for my convenience
df2.rename(columns={"yrsed":"years_school", 
                    "dist":"col_dist", 
                    "dadcoll":"dad_col", 
                    "momcoll":"mom_col"}, inplace=True)

important_statistics = ["mean", "std", "min", "max"]
data_describe2 = df2.describe().loc[important_statistics].transpose().round(2)
data_describe2[important_statistics].to_csv("SS340_HW4_Describe2.csv")
print(data_describe2)    

# =============================================================================
# Initial Regression (Part 2a)
# =============================================================================
print(" Part 2a ".center(100, "="))
result = run_regression(df2.years_school, df2.log_earnings, 
                        xlabel="Years of Schooling",
                        ylabel="log(Earnings)",
                        title="log(Earnings) vs Years of Schooling (Part 2a)",
                        save_title="SS340_HW4_Part2a.png")
model, [beta_0, beta_1], p_val = result

title = "log(Earnings) vs Years of Schooling for Part 2a"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "log(Earnings)", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("log_earnings-years_school beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# 2SLS (Part 2b)
# =============================================================================
print(" Part 2b ".center(100, "="))
years_school = sm.add_constant(df2.years_school)
Z = sm.add_constant(df2.col_dist)
model = IV2SLS(endog=df2.log_earnings, exog=years_school, instrument=Z)
model.cov_type = "HC1"
model = model.fit()
[beta_0, beta_1] = model.params
p_value = model.pvalues[1]

title = "log(Earnings) vs Years of Schooling for Part 2b"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "log_earnings", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("log_earnings-years_school beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Including Dummies (Part 2c1)
# =============================================================================
print(" Part 2c1 ".center(100, "="))
X = df2[["years_school", "mom_col", "dad_col"]]
result = run_regression(X, df2.log_earnings)
model, [beta_0, beta_1, beta_2, beta_3], p_val = result

title = "log(Earnings) vs Years of Schooling for Part 2c1"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "log(Earnings)", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}, {beta_2=:.2f}, {beta_3=:.2f}")
print("log_earnings-years_school beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# Including Dummies IV2SLS (Part 2c2)
# =============================================================================
print(" Part 2c2 ".center(100, "="))

formula = "log_earnings ~ 1 + mom_col + dad_col + [years_school ~ col_dist]"
model = lmIV2SLS.from_formula(formula, data=df2).fit(cov_type="robust")
[beta_0, beta_1, beta_2, beta_3] = model.params
p_value = model.pvalues[1]

title = "log(Earnings) and years of schooling, holding fixed parent's college,\
 for Part 2c2"
normal_summary = model.summary
with open('normal_summary.txt', 'a') as f:
    f.write(title)
    f.write('\n')
    f.write(normal_summary.as_text())
    f.write("\n\n\n")

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("log_earnings-years_school beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# =============================================================================
# First Stage with and Without Parental Controls (Part 2d)
# =============================================================================
print(" Part 2d ".center(100, "="))

# Without Parental Controls
result = run_regression(df2.col_dist, df2.years_school)
model, [beta_0, beta_1], p_val = result

title = "Years of Schooling and Distance from College for Part 2d(1)"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "Years of Schooling", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}")
print("years_school-col_dist beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

# With Parental Controls
X = df2[["col_dist", "mom_col", "dad_col"]]
result = run_regression(X, df2.years_school)
model, [beta_0, beta_1, beta_2, beta_3], p_val = result

title = "Years of Schooling and Distance from College for Part 2d(2)"
normal_summary = get_normal_model_summary(model, title)
stata_summary = get_stata_model_summary(model, "Years of Schooling", title)

print(f"{beta_0=:.2f}, {beta_1=:.2f}, {beta_2=:.2f}, {beta_3=:.2f}")
print("years_school-col_dist beta_1 null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)