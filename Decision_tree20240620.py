#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:56:47 2023

@author: mainaoshige
"""

# Loading the library

# numpy https://numpy.org/ja/
# To install,
# pip install numpy
# or
# conda install numpy
import numpy as np

# pandas https://pandas.pydata.org
# To install,
# pip install pandas
# or
# conda create -c conda-forge -n name_of_my_env python pandas
import pandas as pd

# scikit-learn https://scikit-learn.org/stable/index.html
# To install,
# pip install -U scikit-learn
# or
# conda install scikit-learn-intelex
from sklearn.model_selection import train_test_split

# Matplotlib https://matplotlib.org/3.5.3/index.html
# To install,
# pip install matplotlib
# or
# conda install matplotlib
import matplotlib.pyplot as plt

# To prevent OMP Abort error with anaconda
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

#FileNameOfPatientsData = "covid_data.xlsx"
FileNameOfPatientsData = "covid_data_scored.xlsx"

# List of Items used to analyse
InputItemList = ["Severity","Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph"]
# Using all data
# TrainingItemList = ["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph"]
# Using clinical data
# TrainingItemList = ["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma"]
# Using clinical-blood data
TrainingItemList = ["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma"]

# Read the covid-19 patients Excel data file
allcoviddata_pandas =pd.read_excel(FileNameOfPatientsData)

# Extract data to analyse
coviddata_pandas = allcoviddata_pandas[InputItemList]

# shuffle to make sequence random
#   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
coviddata_pandas_shuffles = coviddata_pandas.sample(frac=1)

#number of test data to extract 
number_of_test = 30
# extract test data
coviddata_test_pandas = coviddata_pandas_shuffles[0:number_of_test]
# extract train and validation data
coviddata_train_pandas = coviddata_pandas_shuffles[number_of_test:]

# dived into data_train, labels_train, pandas -> numpy
data_train_numpy = coviddata_train_pandas[TrainingItemList].values

labels_train_numpy = coviddata_train_pandas['Severity'].values

# without oversamplimg
covid_xtest = coviddata_test_pandas[TrainingItemList].values
covid_ytest = coviddata_test_pandas['Severity'].values

# Split training data and varification data
#X_train,X_test,y_train,y_test = train_test_split(data_train_numpy,labels_train_numpy,test_size = 0.1)

# Create decisiontree
from sklearn import tree

# Set decisiontree
model = tree.DecisionTreeClassifier(max_depth=3)

# Learn model
model.fit(data_train_numpy, labels_train_numpy)

print(model.score(covid_xtest, covid_ytest))

print("ground truth 4 classes", covid_ytest)
y_pred = model.predict(covid_xtest)
print("prediction 4 classes", y_pred)
print("accuracy 4 classes", np.sum(covid_ytest == y_pred) / number_of_test)

covid_ytest = np.where(covid_ytest < 2, 0, 1) 
print("ground truth 2 classes", covid_ytest)
y_pred = np.where(y_pred < 2, 0, 1) 
print("prediction 2 classes", y_pred)
print("accuracy 2 classes", np.sum(covid_ytest == y_pred) / number_of_test)

#print(np.array(y_test))

# Visualization
plt.figure(figsize=(40, 25))
tree.plot_tree(model, feature_names=coviddata_test_pandas.columns.to_list(), class_names=["Mild","ModerateI", "ModerateII", "Severe"],filled=True)

