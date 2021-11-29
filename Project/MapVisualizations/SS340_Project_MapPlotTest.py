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
import json
import pandas as pd

pio.renderers.default = "browser"
# pio.renderers.default = "png"

with open("../Datasets/geojson-counties-fips.json", "r") as f:
    counties = json.load(f)
df = pd.read_csv("../Datasets/CombinedData.csv")
df.fips = df.fips.apply(lambda fip: f"{fip:05d}")

data = df[df.year == 1953][["fips", "tempc", "disasters"]]

fig = px.choropleth(data,
                    geojson=counties,
                    locations='fips',
                    color="tempc",
                    color_continuous_scale="Viridis",
                    range_color=[df.tempc.min(), df.tempc.max()],
                    scope="usa",
                    labels={"tempc": "Temp. degC"},
                    )
fig.update_layout(title="USA Temp. degC in 1953")

fig.write_image("figures/test.png")
fig.show()
