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

def null_hypothesis_test(t_stat, alpha, conf, dof, single_tailed=True):
    """runs the null hypothesis test"""
    t_tab = t_tabulated(conf, dof)
    t_stat = abs(t_stat)
    
    if reject_null(t_stat, t_tab):
        print(f"Reject the null hypothesis with alpha = {alpha*100}%")
    else:
        print(f"Fail to reject the null hypothesis with alpha = {alpha*100}%")
        
    p_value = t.sf(t_stat, dof)
    
    if not single_tailed:
        p_value *= 2
        
    return t_tab, p_value

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
    t_tab, p_val = null_hypothesis_test(t_stat, alpha, conf, dof, False)
    print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")
    
# =============================================================================
# Problem 2
# =============================================================================
def problem2():
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
    
    # fraction smokers = (those who smoke > 0 cigs)/(total number of people)
    frac_smokers =  df["smoker"].mean()
    print("\nPart (b)")
    print(f"Percent smokers: {frac_smokers*100}%")
    
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
    print("Level of education hypothesis test:")
    t_tab, p_val = null_hypothesis_test(t_stat, alpha, conf, dof, False)
    print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")
    
    # hypothesis: level of income is similar for smokers and non-smokers
    # H0: smoker_mean_income - nonsmoker_mean_income = 0
    # Ha: smoker_mean_income - nonsmoker_mean_income != 0   
    alpha = 0.05
    conf = confidence(alpha, single_tailed=False)
    dof = n - 2
    
    # mean-mean null hypothesis
    t_stat, _ = ttest_ind(df["income"][df["smoker"]], 
                          df["income"][~df["smoker"]], 
                          equal_var=False)
    
    print("\nPart (e)")
    print("\nIncome education hypothesis test:")
    t_tab, p_val = null_hypothesis_test(t_stat, alpha, conf, dof, False)
    print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")
    
    # perc_white_smoke = n_white_smokers/n_white
    perc_white_smoke = df["smoker"][df["white"].astype(bool)].mean()
    # perc_nonwhite_smoke = n_nonwhite_smokers/n_nonwhite
    perc_nonwhite_smoke = df["smoker"][~df["white"].astype(bool)].mean()
    print("\nPart (f)")
    print(f"Percent white and smoker: {perc_white_smoke*100:.2f}%")
    print(f"Percent nonwhite and smoker: {perc_nonwhite_smoke*100:.2f}%")
    
    # Cigrarettes column histogram
    plt.figure()
    plt.hist(df["cigs"])
    plt.title("Cigarettes Histogram [Part (g)]")
    plt.xlabel("Cigarettes Smoked Per Day")
    
    plt.ylabel("Count")
    plt.grid(False)
    plt.savefig("SS340_HW1_cigaretteshist.png")
    
    # Cigarettes vs income correlation
    cigs_income_r, p_val = pearsonr(df["cigs"], df["income"])
    print("\nPart (h)")
    print(f"Pearson r btwn cigs per day and income: {cigs_income_r:.3f}")
    if p_val < alpha:
        print("Reject the null hypothesis for cigs-income")
    else:
        print("Fail to reject the null hypothesis for cigs-income")
    print(p_val, alpha)
    
    plt.figure()
    plt.scatter(df["cigs"], df["income"])
    plt.xlabel("Cigarettes Per Day")
    plt.ylabel("Income [$/Year]")
    plt.title("Income vs Cigarettes Smoked Per Day")
    plt.savefig("SS340_HW1_incomevscigs.png")
    plt.show()
    
    # Cigarettes vs Cigarette Price    
    cigs_cigprice_r, p_val = pearsonr(df["cigpric"], df["cigs"])
    print("\nPart (h)")
    print(f"Pearson r btwn cigs per day and cig price: {cigs_cigprice_r:.3f}")
    if p_val < alpha:
        print("Reject the null hypothesis for cigs-cigprice")
    else:
        print("Fail to reject the null hypothesis for cigs-cigprice")
    print(p_val, alpha)
    
    
    
    plt.figure()
    plt.scatter(df["cigs"], df["cigpric"])
    plt.xlabel("Cigarettes Per Day")
    plt.ylabel("Cigarette Price [cents/Pack]")
    plt.title("Cigarette Price vs Cigarettes Smoked Per Day")
    plt.savefig("SS340_HW1_cigpricevscigs.png")
    plt.show()
    
    global local_vars
    local_vars = inspect.currentframe().f_locals

if __name__ == "__main__":
    display_nice("Problem 1", problem1)
    display_nice("Problem 2", problem2)