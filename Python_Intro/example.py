# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
SS340 Cause and Effect
Homework 1
Created: 09/20/2021

Example Python program displaying the following things:
    * Importing libraries
    * Loading a dataset
    * Showing summary statistics
    * Generating the correlation matrix
    * Adding data to the DataFrame
    * t-test
    * Histogram and scatter plots
    * Linear regression
"""
#%% Importing libraries
import numpy as np
from scipy.stats import t
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
from sklearn import linear_model
from sklearn.metrics import r2_score

plt.close("all")    # close any open plots

#%% Load data using Pandas
# can also import csv using pd.read_csv() or excel using pd.read_xlsx()
auto = pd.read_stata("auto.dta")

#%% Summary statistics
# DataFrame.describe() summarizes the DataFrame
auto_describe = auto.describe()
print(auto_describe.round())    # print the summary statistics to the console

#%% Correlation Matrix
# DataFrame.corr() creates a correlation matrix of the DataFrame
corr = auto.corr()
print(corr.round(2))            # print correlation matrix to the console

# Use seaborn to visualize the correlation matrix
plt.figure()
sn.heatmap(corr.round(2), annot=True)
plt.tight_layout()
plt.title("Auto Dataset Correlation Matrix")
plt.show()

#%% Add new data
# Add new data column titled "inefficient" which shows True if mpg < 30
auto["inefficient"] = auto["mpg"] < 30

#%% t-test
# H0: price = 6000
# Ha: price != 6000

H0 = 6000
n = len(auto["price"])      # length of the price vector, i.e. how many prices
dof = n - 1
alpha = 0.05            # significance
conf = 1-alpha/2        # alpha/2 because two-tailed

mean_price = auto["price"].mean()
std_price = auto["price"].std()
SE = std_price/np.sqrt(n)       # standard error

t_stat = abs((mean_price - H0)/SE)
t_tab = t.ppf(conf, dof)
p_val = t.sf(t_stat, dof)*2     # double because two-tailed
print(f"{t_stat=:.2f}, {t_tab=:.2f}, {p_val=:.2f}")

# Test the null hypothesis
if t_stat > t_tab:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")


#%% Plots
# Plot a histogram using matplotlib
plt.figure()
# alternative: auto["price"].hist()
plt.hist(auto["price"])
plt.xlabel("Price")
plt.title("Price Histogram")
plt.show()

# Plot a scatter plot using matplotlib
plt.figure()
# alternative: auto.plot(x="price", y="mpg", style='o')
plt.scatter(auto["price"], auto["mpg"])
plt.xlabel("Price")
plt.ylabel("MPG")
plt.title("Price vs MPG")
plt.show()

# Plot side-by-side plots of foreign vs domestic price vs mpg data
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
mask = auto["foreign"] == "Foreign"     # True-False vector

# foreign plot
ax1.scatter(auto["price"][mask], auto["mpg"][mask])
ax1.set_xlabel("Price")
ax1.set_ylabel("MPG")
ax1.set_title("Price vs MPG of Foreign Cars")

# domestic plot
ax2.scatter(auto["price"][~mask], auto["mpg"][~mask])
ax2.set_xlabel("Price")
ax2.set_title("Price vs MPG of Domestic Cars")
fig.show()

#%% Linear regression
# linear regression of weight on length
lin_regress = linear_model.LinearRegression()   # linear regression model

# reshape the data because sklearn wants column vectors
# -1 means it should decide how many rows, rather than explicitly saying
X = auto["weight"].to_numpy().reshape(-1,1)
y = auto["length"].to_numpy().reshape(-1,1)

# providing the data to be fit
lin_regress.fit(X, y)

# making predictions based on the fit to plot as the regression line
y_pred = lin_regress.predict(X)

# r^2 score metric shows the significance of the fit
r_squared = r2_score(y_pred, y)

# print the coefficients, intercept, and score to the console
print(f"{lin_regress.coef_=}, {lin_regress.intercept_=}, {r_squared=:.3f}")

# plot the data and the fit
plt.figure()
plt.scatter(X, y, label="data")
plt.plot(X, y_pred, 'r', label="fit")   # 'r' sets the line to red
plt.xlabel("Weight")
plt.ylabel("Length")
plt.title("Length vs Weight")
plt.legend()
plt.show()

# multiple linear regression of weight and length on mpg
mult_lin_regress = linear_model.LinearRegression()
X = auto[["weight", "length"]].to_numpy().reshape(-1,2)
y = auto["mpg"].to_numpy().reshape(-1,1)

mult_lin_regress.fit(X, y)

y_pred = mult_lin_regress.predict(X)

r_squared = r2_score(y_pred, y)
coefs = mult_lin_regress.coef_
intercept = mult_lin_regress.intercept_
print(f"{coefs=}, {intercept=}, {r_squared=}")

