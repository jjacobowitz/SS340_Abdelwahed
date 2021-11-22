# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Adding fip codes to disaster dataset, simplifying time to just year, and 
removing unnecessary columns
"""
import pandas as pd

# most of the columns are not needed for our analysis
disaster_data = pd.read_csv("../Datasets/DisasterDeclarationsSummaries.csv",
                            usecols=("state", 
                                     "fyDeclared", 
                                     "incidentType", 
                                     "fipsStateCode", 
                                     "fipsCountyCode",
                                     "designatedArea"))

# combine the state and county fip codes to compare with the temps. dataset
disaster_data["fips"] = ""
for row in disaster_data.itertuples():
    fip = f"{int(row.fipsStateCode):02d}{int(row.fipsCountyCode):03d}"
    disaster_data.at[row.Index, "fips"] = fip
    
# binary data will be easier for applying a disaster to the other dataset
disaster_data["statewide"] = disaster_data["designatedArea"] == "Statewide"

disaster_data.rename(columns={"fyDeclared":"year"}, inplace=True)
disaster_data.to_csv("../Datasets/DisasterData.csv", 
                     columns=("state", 
                              "year", 
                              "incidentType", 
                              "fips", 
                              "statewide"),
                     index=False)
