# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Proof of concept to add value to all rows that meet a criteria. This can be 
used to add that a disaster happened for all counties in a state
"""
import pandas as pd

temps_data = pd.read_csv("TemperatureData.csv")

low = temps_data["tempc"] < 20
temps_data.loc[low, "tempc"] += 100
