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

# import data
if colab:
    # colab needs the google drive mounted
    from google.colab import drive
    drive.mount("/content/drive/")
    source = ("/content/drive/MyDrive/Cooper Union 2021/Cause and Effect/"
              "Final_Project/CombinedData.csv")
else:
    source = "../Datasets/CombinedData.csv"

combined_data = pd.read_csv(source)


# Slice dataframe by the county
i = 10
average_data = []

# Loop through the entire dataset, organizing by state and county
# Alternative is to run two for loops, one for state and one for county
for key, data in combined_data.groupby(["state", "county"]):

  # Determining how many groups to split the sliced dataframe by
  num_groups = np.ceil(data.shape[0]/i)

  # Slicing the dataframe
  result = np.array_split(data, num_groups)

  # key is like ("state", "county"), and is derived from the groupby function
  # This is a better alternative to using x.at[0,"state"] and x.at[0,"county"]
  # Cleaner because we know that everything is the same for those columns
  state, county = key

  # Processing the dataframe
  for x in result:
    
    # We reset index, because otherwise indexing will start at the next group
    # beginning index, i.e. won"t start at zero, which will make the next line
    # not work
    x = x.reset_index(drop = True)

    # Pull out the first and last year, cast them individually, then 
    # concatenate into a single string
    years = f"{x.at[0,'year']}-{x.at[len(x.index)-1,'year']}"

    # Drop year because we don"t need this, we already have the combined years
    # as a string 
    x = x.drop("year", axis=1)

    # Calculate only numberic means (this excludes strings from the operation)
    # note that because FIPS is the same across this dataframe, it is fine to 
    # average it
    row = x.mean(numeric_only=True)
    
    # Append the pulled out years, state and county, giving indexes for the 
    # columns
    row = row.append(pd.Series([years, state, county], index=["year", 
                                                              "state", 
                                                              "county"]))

    # Reorganize data to match original order
    row = row.reindex(index=["fips", 
                             "year", 
                             "tempc", 
                             "county", 
                             "state", 
                             "disasters"])

    # Append processed row into final list (more efficient to append to the list
    # than to a dataframe, so better to convert everything at once to a 
    # dataframe
    average_data.append(row)

# Convert to a dataframe
final_result = pd.DataFrame(average_data)

# Convert to csv
final_result.to_csv("../Datasets/Average_Data.csv", index=False)
