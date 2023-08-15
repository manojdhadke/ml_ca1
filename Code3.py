#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:36:38 2023

@author: Manoj Dhadke
"""

# Loading data libs for data preprocessesing and I/O functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score


russianRealEstate2021Dataset = pd.read_csv('Data/Russia_Real_Estate_2021_Dataset.csv', sep=r',|;', engine='python')
russianRealEstate2021Dataset.head()

#Removes duplicate rows based on all columns.
russianRealEstate2021Dataset.drop_duplicates()

# Rounding off the values
russianRealEstate2021Dataset['area'] = russianRealEstate2021Dataset['area'].astype(int)
russianRealEstate2021Dataset['geo_lat'] = russianRealEstate2021Dataset['geo_lat'].astype(int)
russianRealEstate2021Dataset['geo_lon'] = russianRealEstate2021Dataset['geo_lon'].astype(int)
russianRealEstate2021Dataset['postal_code'] = russianRealEstate2021Dataset['postal_code'].astype(int)

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
russianRealEstate2021Dataset.drop('street_id', inplace=True, axis=1, errors='ignore')
russianRealEstate2021Dataset.drop('id_region', inplace=True, axis=1, errors='ignore')
russianRealEstate2021Dataset.drop('date', inplace=True, axis=1, errors='ignore')

russianRealEstate2021DatasetSplitOne = np.array_split(russianRealEstate2021Dataset, 2)[0]
russianRealEstate2021DatasetSplitTwo = np.array_split(russianRealEstate2021Dataset, 2)[1]

print("R2 Score:", r2_score(russianRealEstate2021DatasetSplitOne, russianRealEstate2021DatasetSplitTwo))


# Split the data into train and test
trainingData, testingData = train_test_split(russianRealEstate2021Dataset, test_size=0.2, random_state=1)
print("Training dataset size:", len(trainingData))
print("Test dataset size:", len(testingData))


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

# Removing the outliers, since data was not needed to drop from lower end this ommiteds
trainingData = trainingData[trainingData['rooms'] > arrayOfUpperBound]
trainingData.shape


testBuildingData = [5, 7, 3, 93.0, 12.0, 55.8821949, 37.264095, 0, 143442.0]

linearRegressionModel = linear_model.LinearRegression()
logisticRegressionModel = LogisticRegression()

linearRegressionModel.fit(trainingData[['level', 'levels', 'rooms', 'area', 'kitchen_area', 'geo_lat', 'geo_lon', 'building_type', 'postal_code']], trainingData['price'])
logisticRegressionModel.fit(trainingData[['level', 'levels', 'rooms', 'area', 'kitchen_area', 'geo_lat', 'geo_lon', 'building_type', 'postal_code']], trainingData['price'])

print("\n Coefficient: ", linearRegressionModel.coef_)
print("\n Intercept: ", linearRegressionModel.intercept_)

# All columns for dataset
# price	level	levels	rooms	area	 kitchen_area	geo_lat	geo_lon	building_type	object_type	postal_code
# Collected from Test Dataset(Sample Data): 5, 7, 3, 93.0, 12.0, 55.8821949, 37.264095, 0, 143442.0
# Linear Regression: 15899434
# Logistic Regression: 2950000
print("\n Original  Price for house is: 11000000")
print("\n Price for house using Linear Regression:", np.floor(linearRegressionModel.predict([testBuildingData])))
print("\n Price for house using Logistic Regression:", np.floor(logisticRegressionModel.predict([testBuildingData])))

print("\nPrediction Model has been successfully finished!")
