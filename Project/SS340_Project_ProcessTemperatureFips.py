# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

This code inputs the county and state values for each observation based on the 
FIP codes
"""
import pandas as pd

# True if importing the blank dataset, False if importing the partially 
# processed data
from_scratch = False

# =============================================================================
# Useful Functions
# =============================================================================
def get_county_state(fip, fips2cands):
    try:
        county = fips2cands[fips2cands.fips == fip]["county"].values[0]
        state = fips2cands[fips2cands.fips == fip]["state"].values[0]
        
        return county, state
    except IndexError:      # fips code could not be found
        return None, None

# =============================================================================
# Import Dataset for fips to county and state lookup
# =============================================================================
# data of the fip code, county, and state
fips_codes = pd.read_csv("Fips2CountyandState.csv")
fips_codes.rename(columns={"FIPS":"fips", "Name":"county", "State":"state"}, 
                  inplace=True)
fips_codes.fips = fips_codes.apply(lambda x: f"{x['fips']:05d}", axis=1)

# =============================================================================
# Import temperature dataset
# =============================================================================
failed = []
# data of the fip code, year, temperature in F, and temperature in C
if from_scratch:
    temps_data = pd.read_csv("CountyYearlyTemperature.csv")
    # blank columns to be filled later
    temps_data[["county", "state"]] = ""
else:
    temps_data = pd.read_csv("TemperatureData.csv")
    with open("failed.txt", "r") as f:
        for row in f.readlines():
            if row != '\n':
                failed.append(row.strip('\n'))
                
temps_data.fips = temps_data.apply(lambda x: f"{int(x['fips']):05d}", axis=1)

# keep track of previous fip (they are sorted) to speed up the process
# print out the fip, county, and state if fails to find in the csv
prev_fip = ""
for row in temps_data.itertuples():
    indx = row.Index
    fip = row.fips
    
    # # skips row if the data is already present
    if not from_scratch and not pd.isnull(row.county):
        continue
    
    if fip != prev_fip:
        county, state = get_county_state(fip, fips_codes)
        prev_fip = fip
    if county is None and fip not in failed:
        failed.append(fip)
        print(f"failed to find {fip=}")
    else:
        temps_data.at[indx, ["county", "state"]] = (county, state)
        
    
    if indx%1000 == 0:
        print(indx)
        
# =============================================================================
# Save results after filling in the county and state data
# =============================================================================
print(indx)
temps_data.to_csv("TemperatureData.csv", index=False)
with open("failed.txt", "w") as f:
    for fail in failed:
        f.write(fail+'\n')
    
