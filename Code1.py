#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 00:44:08 2023

Delete Rows
This is been used to remove excess data from file to make it lightweight.
It was necessory to work on our local system as well as to enject data in our ML model

@author: Manoj Dhadke
"""

import pandas as pd

# The dataset contains 11.358.150 real estate objects in Russia.
russianRealEstate2021Dataset = pd.read_csv('Data/Russia_Real_Estate_2021_Dataset_Complete.csv', sep=';')

# Since it was very much heavy file(891.06 MB), had to reduce size to make it work on local system. In addition to this, to make it easiler to make distribution of training, validation and test data
# 11258149 => 100000
# 11358149 => 20000
# 11348149 => 10000
# 11357149 => 1000
russianRealEstate2021Dataset = russianRealEstate2021Dataset.iloc[11357149:-1]
russianRealEstate2021Dataset.interpolate(method ='linear', limit_direction ='forward', inplace=True)

russianRealEstate2021Dataset.to_csv('Data/Russia_Real_Estate_2021_Dataset.csv', index=False)

print(russianRealEstate2021Dataset)
