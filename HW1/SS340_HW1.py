# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
SS340 Cause and Effect
Homework 1
Created: 09/23/2021
Due: 10/04/2021

Python program for Homework 1
"""
import numpy as np
from scipy.stats import t, ttest_ind, pearsonr
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

plt.close("all")    # close any currently open plots

# =============================================================================
# Helper Functions
# =============================================================================


def confidence(alpha, single_tailed=True):
    """calculates the single- or double-tailed confidence"""
    if single_tailed:
        return 1-alpha
    else:
        return 1-(alpha/2)


def standard_error(std, n):
    """calculates the standard error"""
    return std/np.sqrt(n)


def t_statistic(mean, H0, SE):
    """calculates the t-statistic"""
    return abs((mean-H0)/SE)


def t_tabulated(conf, dof):
    """calculates the tabulated t-statistic"""
    return t.ppf(conf, dof)


def p_value(t_stat, dof, single_tailed=True):
    """calculates the p-value"""
    p_val = t.sf(t_stat, dof)

    # double the p-value if two-tailed
    if not single_tailed:
        p_val *= 2

    return p_val


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


def find_nearest(a, a0):
    """Element in nd array `a` closest to the scalar value `a0`.
    modified from: https://stackoverflow.com/a/10465997/12131013
    """
    idx = np.abs(a - a0).argmin()
    value = a.flat[idx]
    return idx, value


# =============================================================================
# Problem 1
# =============================================================================
print("Problem 1".center(80, "~"))

n = 150         # sample size
mean = 3.2      # sample mean GPA
std = 0.5       # sample standard deviation
H0 = 3.5        # null hypothesis
alpha = 0.1     # 10% significance

conf = confidence(alpha, single_tailed=False)   # confidence
SE = standard_error(std, n)                     # standard error
t_stat = t_statistic(mean, H0, SE)              # t-statistic
dof = n - 1                                     # degrees of freedom
p_val = p_value(t_stat, dof, single_tailed=False)

# if p_val <= alpha, then we reject the null hypothesis
null_hypothesis_test(p_val, alpha)
t_tab = t_tabulated(conf, dof)
print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")

# Plotting
rv = t(df=dof, loc=0, scale=1)
x = np.linspace(rv.ppf(0.0001), rv.ppf(0.9999), 1000)
y = rv.pdf(x)

indx_right, _ = find_nearest(y, rv.pdf(t_tab))
indx_left, _ = find_nearest(y[:len(y)//2], rv.pdf(-t_tab))  # avoid right sol.

plt.figure()
plt.plot(x, y, label="t-distribution")
plt.fill_between(np.linspace(t_tab, x[-1], len(y)-indx_right), y[indx_right:],
                 alpha=0.5, color="tab:blue", label="rejection region")
plt.fill_between(np.linspace(x[0], -t_tab, indx_left), y[:indx_left],
                 alpha=0.5, color="tab:blue")

plt.title(f"t-Distribution Plot\n(dof = {dof})")
plt.legend()
plt.show()
plt.savefig("figures/SS340_HW1_problem1.png")

# =============================================================================
# Problem 2
# =============================================================================
print("Problem 2".center(80, "~"))
# import the csv data
df = pd.read_csv("Wooldrridge smoking data.csv")
n = len(df)     # number of individuals
cols = df.columns
cols_exclude_personid = set(cols)-{"personid"}

# dfs for smoker and non-smoker
# if they smoke any number of cigarettes, then they are a smoker
df["smoker"] = df['cigs'] > 0

# summary statistics of the data
describe = df.describe().round(2)
print("\nPart (a)")
print(describe[cols_exclude_personid])

# percent smokers = (those who smoke > 0 cigs)/(total number of people)
perc_smokers = df["smoker"].mean()
print("\nPart (b)")
print(f"Percent smokers: {perc_smokers*100}%")

# describing smokers and nonsmokers separately
describe_smokers = df[df["smoker"]].describe().round(2)
print("\nPart (c)")
print("Smokers:")
print(describe_smokers[cols_exclude_personid])

describe_nonsmokers = df[~df["smoker"]].describe().round(2)
print("Non-smokers:")
print(describe_nonsmokers[cols_exclude_personid])

# hypothesis: level of education is similar for smokers and non-smokers
# H0: smoker_mean_education - nonsmoker_mean_education = 0
# Ha: smoker_mean_education - nonsmoker_mean_education != 0
alpha = 0.05
conf = confidence(alpha, single_tailed=False)
dof = n - 2

# mean-mean null hypothesis
t_stat, p_val = ttest_ind(df["educ"][df["smoker"]],
                          df["educ"][~df["smoker"]],
                          equal_var=False)

print("\nPart (d)")
print("education-smoker null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)
t_tab = t_tabulated(conf, dof)
print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")

# hypothesis: level of income is similar for smokers and non-smokers
# H0: smoker_mean_income - nonsmoker_mean_income = 0
# Ha: smoker_mean_income - nonsmoker_mean_income != 0
alpha = 0.05
conf = confidence(alpha, single_tailed=False)
dof = n - 2     # -2 because there are two means used

# mean-mean null hypothesis
t_stat, p_val = ttest_ind(df["income"][df["smoker"]],
                          df["income"][~df["smoker"]],
                          equal_var=False)

print("\nPart (e)")
print("income-smoker null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)
t_tab = t_tabulated(conf, dof)
print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")

# perc_white_smoke = n_white_smokers/n_white
perc_white_smoker = df["smoker"][df["white"].astype(bool)].mean()
# perc_nonwhite_smoke = n_nonwhite_smokers/n_nonwhite
perc_nonwhite_smoker = df["smoker"][~df["white"].astype(bool)].mean()
print("\nPart (f)")
print(f"Percent white and smoker: {perc_white_smoker*100:.2f}%")
print(f"Percent nonwhite and smoker: {perc_nonwhite_smoker*100:.2f}%")

# Hypothesis testing the difference between proportions
# H0: perc_white_smoker = perc_nonwhite_smoker
# Ha: perc_white_smoker != perc_nonwhite_smoker
# n_white: number of white people
# n_nonwhite: number of nonwhite people
# n_white_smoker: number of white smokers
# n_nonwhite_smoker: number of nonwhite smokers
n_white = np.sum(df.white.astype(bool))
n_nonwhite = np.sum(~df.white.astype(bool))
n_white_smoker = np.sum(df.smoker[df.white.astype(bool)])
n_nonwhite_smoker = np.sum(df.smoker[~df.white.astype(bool)])
z_val, p_val = proportions_ztest(np.array([n_white_smoker,
                                           n_nonwhite_smoker]),
                                 np.array([n_white,
                                           n_nonwhite]))
print(f"{z_val=:.2f},{p_val=:.2f}")
print("white-nonwhite smoker null hypothesis test:", end=" ")
null_hypothesis_test(p_val, 0.05)

# Cigrarettes column histogram
plt.figure()
plt.hist(df["cigs"], ec='k')
plt.title("Cigarettes Histogram")
plt.xlabel("Cigarettes Smoked Per Day")
# turn on minor ticks for both axes
plt.minorticks_on()
# turn off y-axis minor ticks
plt.gca().yaxis.set_tick_params(which='minor', bottom=False)

plt.ylabel("Count")
plt.grid(False)
plt.savefig("figures/SS340_HW1_cigaretteshist.png")

# Cigarettes vs income correlation
cigs_income_r, p_val = pearsonr(df["cigs"], df["income"])
print("\nPart (h)")
print(f"Pearson r btwn cigs per day and income: {cigs_income_r:.3f}")
print("cigs-income null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

plt.figure()
plt.scatter(df["cigs"], df["income"])
plt.xlabel("Cigarettes Smoked Per Day")
plt.ylabel("Income [$/Year]")
plt.title("Income vs Cigarettes Smoked Per Day")
plt.savefig("figures/SS340_HW1_incomevscigs.png")
plt.show()

# Cigarettes vs Cigarette Price
cigs_cigprice_r, p_val = pearsonr(df["cigpric"], df["cigs"])
print("\nPart (h)")
print(f"Pearson r btwn cigs per day and cig price: {cigs_cigprice_r:.3f}")
print("cigs-cigprice null hypothesis test:", end=" ")
null_hypothesis_test(p_val, alpha)

plt.figure()
plt.scatter(df["cigs"], df["cigpric"])
plt.xlabel("Cigarettes Smoked Per Day")
plt.ylabel("Cigarette Price [cents/Pack]")
plt.title("Cigarette Price vs Cigarettes Smoked Per Day")
plt.savefig("figures/SS340_HW1_cigpricevscigs.png")
plt.show()
