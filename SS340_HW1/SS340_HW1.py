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
from scipy.stats import t
import pandas as pd
import matplotlib.pyplot as plt
import inspect

local_vars = {}

plt.close("all")    # close any currently open plots

# =============================================================================
# Helper Functions
# =============================================================================
def display_nice(title, function):
    """Adds dividers between problem sections in the print outs"""
    print(title)
    print()
    function()
    print("~"*80)
    
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

def reject_null(t_statistic, t_tabulated):
    """returns True if the null hypothesis is rejected, False if it fails to be
    rejected
    """
    return t_statistic > t_tabulated

def null_hypothesis_test(t_stat, alpha, conf, dof):
    """runs the null hypothesis test"""
    t_tab = t_tabulated(conf, dof)
    
    if reject_null(t_stat, t_tab):
        print(f"Reject the null hypothesis with alpha = {alpha*100}%")
    else:
        print(f"Fail to reject the null hypothesis with alpha = {alpha*100}%")

# =============================================================================
# Problem 1
# =============================================================================
def problem1():
    n = 150         # sample size
    mean = 3.2      # sample mean GPA
    std = 0.5       # sample standard deviation
    H0 = 3.5        # null hypothesis
    alpha = 0.1     # 10% significance
    
    # calculate the confidence
    conf = confidence(alpha, single_tailed=False)
    
    SE = standard_error(std, n)             # standard error
    t_stat = t_statistic(mean, H0, SE)      # t-statistic
    dof = n - 1                             # degrees of freedom
    
    # if t-statistic > t-tabulated then we reject the null hypothesis
    null_hypothesis_test(t_stat, alpha, conf, dof)
    
# =============================================================================
# Problem 2
# =============================================================================
def problem2():
    # import the csv data
    df = pd.read_csv("Wooldrridge smoking data.csv")
    n = len(df)     # number of individuals
    
    # dfs for smoker and non-smoker
    # if they smoke at least 1 cigarette, they are a smoker
    mask = df['cigs'] > 0
    smokers = df[mask]
    nonsmokers = df[~mask]
    
    # number of smokers and nonsmokers
    n_smokers = len(smokers)
    n_nonsmokers = len(nonsmokers)
    
    # summary statistics of the data
    describe = df.describe()
    print(describe)
    
    # fraction smokers = (those who smoke > 0 cigs)/(total number of people)
    frac_smokers =  n_smokers/n
    print(f"Percent smokers: {frac_smokers*100}%")
    
    # describing smokers and nonsmokers separately
    describe_smokers = smokers.describe()
    print("Smokers:")
    print(describe_smokers)
    
    describe_nonsmokers = nonsmokers.describe()
    print("Non-smokers:")
    print(describe_nonsmokers)
    
    # hypothesis: level of education is similar for smokers and non-smokers
    # H0: smoker_mean_education - nonsmoker_mean_education = 0
    # Ha: smoker_mean_education - nonsmoker_mean_education != 0
    # For brevity, "smoker" was shortened to "s" and "nonsmoker" to "ns"
    s_mean_educ = smokers["educ"].mean()
    s_std_educ = smokers["educ"].std()
    ns_mean_educ = nonsmokers["educ"].mean()
    ns_std_educ = nonsmokers["educ"].std()
    
    alpha = 0.05
    conf = confidence(alpha, single_tailed=False)
    dof = n - 2
    
    # mean-mean null hypothesis
    # source: http://www.stat.yale.edu/Courses/1997-98/101/meancomp.htm
    t_stat = (s_mean_educ - ns_mean_educ)/np.sqrt(s_std_educ**2/n_smokers + ns_std_educ**2/n_nonsmokers)
    
    print("Level of education hypothesis test:")
    null_hypothesis_test(t_stat, alpha, conf, dof)
    
    # hypothesis: level of income is similar for smokers and non-smokers
    # H0: smoker_mean_income - nonsmoker_mean_income = 0
    # Ha: smoker_mean_income - nonsmoker_mean_income != 0
    # For brevity, "smoker" was shortened to "s" and "nonsmoker" to "ns"
    s_mean_inc = smokers["income"].mean()
    s_std_inc = smokers["income"].std()
    ns_mean_inc = nonsmokers["income"].mean()
    ns_std_inc = nonsmokers["income"].std()
    
    alpha = 0.05
    conf = confidence(alpha, single_tailed=False)
    dof = n - 2
    
    t_stat = ((s_mean_inc - ns_mean_inc)
              /np.sqrt(s_std_inc**2/n_smokers + ns_std_inc**2/n_nonsmokers))
    
    print("\nIncome education hypothesis test:")
    null_hypothesis_test(t_stat, alpha, conf, dof)
    
    # use the data to show that smokers are more commonly non-white vs white
    n_white = len(df.white[df.white == 1])
    n_nonwhite = n - n_white
    mask_white = smokers.white == 1
    n_white_smokers = len(smokers[mask_white])
    n_nonwhite_smokers = len(smokers[~mask_white])
    
    perc_white_smoke = n_white_smokers/n_white
    perc_nonwhite_smoke = n_nonwhite_smokers/n_nonwhite
    print(f"\nPercent white and smoker: {perc_white_smoke*100:.2f}%")
    print(f"Percent nonwhite and smoker: {perc_nonwhite_smoke*100:.2f}%")
    
    # Cigrarettes column histogram
    df["cigs"].hist()
    plt.title("Cigarettes Histogram")
    plt.xlabel("Cigarettes Smoked Per Day")
    
    plt.ylabel("Count")
    plt.grid(False)
    plt.savefig("SS340_HW1_cigaretteshist.png")
    
    # Cigarettes vs income correlation
    cigs_income_corr = df["cigs"].corr(df["income"])
    print(f"\nCorr btwn cigs per day and income: {cigs_income_corr:.3f}")

    
    # Cigarettes vs Cigarette Price
    cigs_cigprice_corr = df["cigpric"].corr(df["cigs"])
    print(f"Corr btwn cigs per day and cig price: {cigs_cigprice_corr:.3f}")
    
    global local_vars
    local_vars = inspect.currentframe().f_locals

if __name__ == "__main__":
    display_nice("Problem 1", problem1)
    display_nice("Problem 2", problem2)