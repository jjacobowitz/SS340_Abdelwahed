# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:55:41 2021

@author: Jared
"""
import pandas as pd
from geopy.geocoders import Nominatim

def get_latlong(county, state, country="US"):
    geolocator = Nominatim(user_agent="my_user_agent")
    
    # for some reason, names cannot have spaces
    loc = geolocator.geocode(f"{county.replace(' ','')},{state},{country}")

    return loc.latitude, loc.longitude

def get_county_state(fip, fips2cand2):
    county = fips2cand2[fips2cand2.fips == fip]["county"].values[0]
    state = fips2cand2[fips2cand2.fips == fip]["state"].values[0]
    
    return county, state

# data of the fip code, county, and state
fips2cand2 = pd.read_csv("Fips2CountyandState.csv", dtype=str)
fips2cand2.rename(columns={"FIPS":"fips", "Name":"county", "State":"state"}, 
                  inplace=True)

# data of the fip code, year, temperature in F, and temperature in C
data = pd.read_csv("CountyYearlyTemperature.csv")

# temporarily filling with blank data to be replaced later
data[["county", "state"]] = ""
data[["lat", "long"]] = 0., 0.

prev_fip = ""
for indx, row in data.iterrows():
    fip = str(int(row.fips))
    if fip != prev_fip:
        county, state = get_county_state(fip, fips2cand2)
        try:
            lat, long = get_latlong(county, state)
        except AttributeError:
            lat, long = None, None
            print(fip, county, state)
        prev_fip = fip
    elif lat is not None:
        data.at[indx, ["county", "state", "lat", "long"]] = (county, 
                                                             state, 
                                                             lat, 
                                                             long)
    
data.csv("TemperatureData.csv")
    
