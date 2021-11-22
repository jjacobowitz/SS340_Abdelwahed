# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:55:41 2021

@author: Jared
"""
import pandas as pd
from geopy.geocoders import Nominatim

# based on: https://stackoverflow.com/a/54965835/12131013
def get_latlong(county, state, country="US"):
    geolocator = Nominatim(user_agent="my_user_agent")
    
    # for some reason, names cannot have spaces
    loc = geolocator.geocode(f"{county.replace(' ','')},{state},{country}")

    return loc.latitude, loc.longitude

def get_county_state(fip, fips2cands):
    if fips2cands[fips2cands.fips == fip].size == 0:
        return "",""
    county = fips2cands[fips2cands.fips == fip]["county"].values[0]
    state = fips2cands[fips2cands.fips == fip]["state"].values[0]
    
    return county, state

# data of the fip code, county, and state
fips2cands = pd.read_csv("../Datasets/Fips2CountyandState.csv")
fips2cands.rename(columns={"FIPS":"fips", "Name":"county", "State":"state"}, 
                  inplace=True)
# fips2cands.fips = fips2cands.fips.astype(str)

# data of the fip code, year, temperature in F, and temperature in C
data = pd.read_csv("../Datasets/TemperatureData.csv")
failed = []

# keep track of previous fip (they are sorted) to speed up the process
# print out the fip, county, and state if get_longlat fails
prev_fip = ""
for row in data.itertuples():
    indx = row.Index
    fip = int(row.fips)
    # these fail for some reason
    if fip in [5059, 5081, 9009, 17161, 18089, 19021, 22075]:           
        continue
    # skip already filled rows
    if not pd.isnull(row.county) and row.county != "":
        continue
    if fip != prev_fip:
        county, state = get_county_state(fip, fips2cands)
        # if county
        try:
            lat, long = get_latlong(county, state)
        except AttributeError:
            # this happens when get_latlong fails to find the location
            print(indx, fip, county, state)
            failed.append((indx, fip, county, state))
            lat, long = None, None
        # geopy has a custom timeout error
        # this catches that and saves the data so the code can be run again
        except Exception as e:
            print(e)
            break
        prev_fip = fip
    if lat is not None:
        data.at[indx, ["county", "state", "lat", "long"]] = (county, 
                                                              state, 
                                                              lat, 
                                                              long)
print(indx)
data.to_csv("../Datasets/TemperatureData.csv", index=False)
with open("../Datasets/failed.txt", "w") as f:
    f.write("indx,fip,county,state\n")
    for fail in failed:
        f.write(",".join(str(a) for a in fail)+'\n')
    
