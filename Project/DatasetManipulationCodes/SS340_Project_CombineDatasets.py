# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Combine the temperature and disaster datasets
"""
import pandas as pd
# import numpy as np

disaster_data = pd.read_csv("../Datasets/DisasterData.csv")
temps_data = pd.read_csv("../Datasets/TemperatureData.csv")
temps_data.dropna(inplace=True)

# max of mins is the smallest common value
# min of maxes is the largest common value
min_year = max(min(disaster_data["year"]), min(temps_data["year"]))
max_year = min(max(disaster_data["year"]), max(temps_data["year"]))

temps_data = temps_data[(temps_data["year"] <= max_year)
                        & (temps_data["year"] >= min_year)]
disaster_data = disaster_data[(disaster_data["year"] <= max_year)
                              & (disaster_data["year"] >= min_year)]

temps_data["disasters"] = 0

for row in disaster_data.itertuples():
    year = row.year
    indx = temps_data["year"] == year
    if row.statewide:
        indx &= (temps_data["state"] == row.state)
    else:
        indx &= (temps_data["fips"] == row.fips)
    temps_data.loc[indx, "disasters"] += 1
    
temps_data.to_csv("../Datasets/CombinedData.csv", index=False)