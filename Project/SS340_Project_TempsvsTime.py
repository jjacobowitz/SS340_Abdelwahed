# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Plot county temperatures vs time to see if there is a trend
"""
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close("all")

temps_data = pd.read_csv("TemperatureData.csv")

countyandstate = temps_data[["county","state"]].drop_duplicates().reset_index()

fig, ax = plt.subplots()

prev_state = ""
for row in countyandstate.itertuples():
    county = row.county
    state = row.state
    if state != prev_state:
        state_indx = (temps_data.state == state)
    indx = ((temps_data.county == county) & state_indx)
    ax.plot(temps_data.year[indx], temps_data.tempc[indx])
    if not row.Index % 1000:
        print(row.Index)