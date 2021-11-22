# -*- coding: utf-8 -*-
"""
Created by: Josh Yoon and Derek Lee
Edited by: Jared Jacobowitz
Fall 2021
SS340 Cause and Effect

Code originally written in Google Colab, link below
https://colab.research.google.com/drive/1bCdpUFBedhmbF1YvKKJNrPVGnOEkLvEh

This code aggregates the years to observe trends over year ranges.
"""
import pandas as pd
import numpy as np

colab = False

# select proper data source
if colab:
    # colab needs the google drive mounted
    from google.colab import drive
    drive.mount("/content/drive/")
    source = ("/content/drive/MyDrive/Cooper Union 2021/Cause and Effect/"
              "Final_Project/CombinedData.csv")
else:
    source = "CombinedData.csv"

# import the data
combined_data = pd.read_csv(source)

# Slice dataframe by the county
max_group_size = 10

# Determining how many groups to split the sliced dataframe by
number_of_years = combined_data.year.max() - combined_data.year.min() + 1
num_groups = np.ceil(number_of_years/max_group_size)


average_data = []

# Loop through the entire dataset, organizing by state and county
# Alternative is to run two for loops, one for state and one for county
for _, county_data in combined_data.groupby("fips"):
    fips = county_data.fips.iloc[0]
    county = county_data.county.iloc[0]
    state = county_data.state.iloc[0]

    # slicing the dataframe into the groups
    result = np.array_split(county_data, num_groups)
  
    # average the data from each group and append to the array
    for group in result:
        # new "years" variable is a string of the group's start and end years
        years = f"{group.year.min()}-{group.year.max()}"
        
        # numeric_only only averages numeric data
        group_mean = group.mean(numeric_only=True)
        
        # append data to the main list
        average_data.append([fips,
                             years, 
                             group_mean.tempc,
                             county, 
                             state, 
                             group_mean.disasters])
        
# convert to a dataframe
final_result = pd.DataFrame(average_data, columns=["fips",
                                                   "years",
                                                   "tempc",
                                                   "county", 
                                                   "state", 
                                                   "disasters"])

# Save csv
final_result.to_csv("Average_Data.csv", index=False)
