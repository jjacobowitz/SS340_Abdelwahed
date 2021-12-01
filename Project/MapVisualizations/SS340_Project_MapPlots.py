# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Creating a choropleth map for the report
"""
import plotly.express as px
import plotly.io as pio
import numpy as np
import json
import pandas as pd

pio.renderers.default = "browser"
# pio.renderers.default = "png"

with open("../Datasets/geojson-counties-fips.json", "r") as f:
    counties = json.load(f)
df = pd.read_csv("../Datasets/CombinedData.csv")
df.fips = df.fips.apply(lambda fip: f"{fip:05d}")
df.disasters = np.log(df.disasters)
df.disasters[df.disasters == -np.inf] = 0

range_color = {"tempc": [df.tempc.min(), df.tempc.max()],
               "disasters": [df.disasters.min(), df.disasters.max()]}

for year in set(df.year):
    print(year)

    data = df[df.year == year][["fips", "tempc", "disasters"]]

    plots = {"tempc": "Temp. degC",
             "disasters": "ln(Disasters)"}
    for key, value in plots.items():
        fig = px.choropleth(data,
                            geojson=counties,
                            locations='fips',
                            color=key,
                            color_continuous_scale=("Bluered"
                                                    if key == "tempc"
                                                    else "Viridis"),
                            range_color=range_color[key],
                            scope="usa",
                            labels={key: value},
                            )
        fig.update_layout(title=f"USA {value} in {year}")

        fig.write_image(f"figures/SS340_map_{year}{key}.png")
        # fig.show()
