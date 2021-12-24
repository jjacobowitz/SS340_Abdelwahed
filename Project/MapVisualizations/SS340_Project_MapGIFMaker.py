# -*- coding: utf-8 -*-
"""
Jared Jacobowitz
Fall 2021
SS340 Cause and Effect
Final Project

Create gifs of the tempc and disasters US data
"""
import imageio
import glob

for end in ["tempc", "disasters"]:
    filenames = []
    for file in glob.glob(f"figures/*{end}.png"):
        filenames.append(file)

    # copied from: https://stackoverflow.com/a/35943809/12131013
    images = []
    for filename in filenames:
        if "2019" in filename:
            repeated = [imageio.imread(filename) for _ in range(10)]
            images.extend(repeated)
        images.append(imageio.imread(filename))
    kwargs = {'duration': 0.1}     # frame duration in seconds
    imageio.mimsave(f'figures/SS340_map_{end}.gif', images, **kwargs)
