#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:37:33 2023

@author: Manoj Dhadke
"""

# Loading data libs for data preprocessesing and I/O functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# For visualizations
from matplotlib import pyplot as plt
#%matplotlib inline # to draw the plots immediately after the current cell
import seaborn as sns


russianRealEstate2021Dataset = pd.read_csv('Data/Russia_Real_Estate_2021_Dataset.csv', sep=',')
russianRealEstate2021Dataset.head()

#Removes duplicate rows based on all columns.
russianRealEstate2021Dataset.drop_duplicates()

# Using existing values in the Russian Real Estate Dataset to fill the missing values
# To ignore any errors like - TypeError: Cannot interpolate with all object-dtype columns in the DataFrame. Try setting at least one column to a numeric dtype.
try:
    russianRealEstate2021Dataset.interpolate(method ='linear', limit_direction ='forward', inplace=True)
except ValueError:
    print("Please ignore this error!")
except TypeError:
    print("Please ignore this error!")
# Removing the house_id column, because it is of no use in our prediction
# del russianRealEstate2021Dataset['house_id'] #OR
russianRealEstate2021Dataset.drop('house_id', inplace=True, axis=1, errors='ignore')
russianRealEstate2021Dataset.drop('kitchen_area', inplace=True, axis=1, errors='ignore')
russianRealEstate2021Dataset.drop('street_id', inplace=True, axis=1, errors='ignore')
russianRealEstate2021Dataset.drop('id_region', inplace=True, axis=1, errors='ignore')

# Split the data into train and test
trainingData, testingData = train_test_split(russianRealEstate2021Dataset, test_size=0.2, random_state=1)
print("Training dataset size:", len(trainingData))
print("Test dataset size:", len(testingData))

fig, ax = plt.subplots(2, 3, figsize=(20, 9))
ax = ax.flatten()

sns.set_theme(style="whitegrid", palette="pastel")

trainingData['Time'] = np.arange(len(trainingData.date))

#Calculating IQR
quartile1 = np.percentile(trainingData['rooms'], 25, interpolation = 'midpoint')
quartile3 = np.percentile(trainingData['rooms'], 75, interpolation = 'midpoint')
iqr = quartile3 - quartile1
 
# Upper bound
upperBound = quartile3+1.5*iqr
arrayOfUpperBound=np.array(trainingData['rooms'] >= upperBound)
# Lower bound
lowerBound=quartile1-1.5*iqr
arrayOfLowerBound=np.array(trainingData['rooms'] <= lowerBound)

# Removing the outliers, since data was not needed to drop from lower end this ommited
trainingData = trainingData[trainingData['rooms'] > arrayOfUpperBound]
trainingData.shape

# #To show heatmap of correlation
# sns.heatmap(trainingData.corr(), vmin=-1, vmax=1,
# annot=True,cmap="mako")
# plt.show()

sns.lineplot(data=trainingData, x="Time", y="price", ax=ax[0])
ax[0].set_title("Price vs Date")
sns.lineplot(data=trainingData, x="level", y="price", ax=ax[1])
ax[1].set_title("Price vs Level")
sns.lineplot(data=trainingData, x="levels", y="price", ax=ax[2])
ax[2].set_title("Price vs Levels")
sns.lineplot(data=trainingData, x="rooms", y="price", ax=ax[3])
ax[3].set_title("Price vs Rooms")
sns.lineplot(data=trainingData, x="area", y="price", ax=ax[4])
ax[4].set_title("Price vs Area")

sns.lineplot(data=trainingData, x="building_type", y="price").set_title("Price vs Building Type")
sns.lineplot(data=trainingData, x="geo_lat", y="price").set_title("Price vs Latitude")
sns.lineplot(data=trainingData, x="geo_lon", y="price").set_title("Price vs Longitude")
sns.lineplot(data=trainingData, x="object_type", y="price").set_title("Price vs Object Type")
sns.lineplot(data=trainingData, x="postal_code", y="price", ax=ax[5])
ax[5].set_title("Price vs Postal Code")

# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show();
 
print(trainingData.corr())
print("\nPreprocessing have been successfully done!")
