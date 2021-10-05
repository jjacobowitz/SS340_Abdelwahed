# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
SS340 Cause and Effect
Homework 1
Created: 10/01/2021
Due: 10/04/2021

Python program to create the plot for HW 1 problem 1
"""
from scipy.stats import t
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

# modified from: https://stackoverflow.com/a/10465997/12131013
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`."
    idx = np.abs(a - a0).argmin()
    value = a.flat[idx]
    return idx, value


n = 150
dof = n - 1

mean = 3.2      # sample mean GPA
std = 0.5       # sample standard deviation
H0 = 3.5        # null hypothesis
alpha = 0.10
conf = 1 - alpha/2

t_tab = t.ppf(conf, dof)

SE = std/np.sqrt(n)
t_stat = abs((mean-H0)/SE)

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
# plt.vlines(t_stat, 0, 0.4, 'r', label="t-stat")
plt.title(f"t-Distribution Plot\n(dof = {dof})")
plt.legend()
plt.show()
plt.savefig("SS340_HW1_problem1.png")