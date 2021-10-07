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
import matplotlib.pyplot as plt

# close any currently open plots
plt.close("all")

def run_regression(x, y, xlabel, ylabel, title, save_title):
    x_sm = sm.add_constant(x)
    model = sm.OLS(y, x_sm).fit()
    print(model.summary())
    
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

# save and print the summary statistics of the data
data_19_describe = data_19[summary_cols].describe().round(2)
print(data_19_describe)

# high-income dummy variable. high-income is defined as income > median income
data_19["hiincome"] = data_19["fincbtxm"] > data_19["fincbtxm"].median()

# total expenditure variable, sum of CQ and PQ expenditures
data_19["totexp"] = data_19[["totexpcq", "totexppq"]].sum(axis=1)*4

# linear regression: all
run_regression(data_19["fincbtxm"].to_numpy(),
               data_19["totexp"].to_numpy(),
               "Income [$/yr.]",
               "Total Expenditure [$/qtr.]",
               "Expenditure vs Income for All Data",
               "SS340_HW1_expvsincall.png")

# linear regression: high-income
run_regression(data_19["fincbtxm"][data_19["hiincome"]].to_numpy(),
               data_19["totexp"][data_19["hiincome"]].to_numpy(),
               "Income [$/yr.]",
               "Total Expenditure [$/qtr.]",
               "Expenditure vs Income for High-Income HHs",
               "SS340_HW1_expvsinchiincome.png")

# linear regression: low-income
run_regression(data_19["fincbtxm"][~data_19["hiincome"]].to_numpy(),
               data_19["totexp"][~data_19["hiincome"]].to_numpy(),
               "Income [$/yr.]",
               "Total Expenditure [$/qtr.]",
               "Expenditure vs Income for Low-Income HHs",
               "SS340_HW1_expvsincloincome.png")