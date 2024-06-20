# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:55:06 2023

@author: Yuki Sumi

"""

# Loading the library

# pandas https://pandas.pydata.org
# pip install pandas
# or
# conda create -c conda-forge -n name_of_my_env python pandas
import pandas as pd


# openpyxl https://pypi.org/project/openpyxl/
# To install,
# pip install openpyxl==3.1.2

# numpy https://numpy.org/ja/
# pip install numpy
# or
# conda install numpy
import numpy as np

# seaborn https://seaborn.pydata.org/index.html
# pip install seaborn
# or
# conda install seaborn -c conda-forge
# import seaborn as sns

# matplotlib https://matplotlib.org/stable/
# pip install matplotlib
# or
# conda install -c conda-forge matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Loading data
# file name of covid-19 patients' data
FileNameOfPatientsData = "covid_data.xlsx"
FileNameOfNormalizedPatientsData = "covid_data_scored.xlsx"

# List of Items used to analyse
InputItemList = ["Severity","Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph"]
# Using all data
AnalyzeItemList = ["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph"]

# Read the covid-19 patients Excel data file
allcoviddata_pandas = pd.read_excel(FileNameOfPatientsData)
normalized_allcoviddata_pandas = allcoviddata_pandas.copy(deep=True)
# Extract data to analyse
#coviddata_pandas = allcoviddata_pandas[AnalyzeItemList]
#print("coviddata_pandas", coviddata_pandas)

# scoring

before_normalization = allcoviddata_pandas["Urea_nitrogen"]
def Urea_nitrogen_func(x):
    if x < 22:
        return 0
    elif x <= 33:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(Urea_nitrogen_func)
normalized_allcoviddata_pandas["Urea_nitrogen"] = after_normalization


before_normalization = allcoviddata_pandas["Lymphocyte"]
def Lymphocyte_func(x):
    if x < 750:
        return 0
    elif x <= 1500:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(Lymphocyte_func)
normalized_allcoviddata_pandas["Lymphocyte"] = after_normalization


number_of_patients = len(before_normalization)
for i in range (number_of_patients):
    print(i)
    # print(before_normalization.iloc[i ,])
    gender = allcoviddata_pandas.iloc[i ,]["Gender"]
    creatinine = allcoviddata_pandas.iloc[i ,]["Creatinine"]

    if gender == 0: # Female
        if creatinine < 0.46:
            normalizeddata = 0
        elif creatinine <= 0.79:
            normalizeddata = 1
        else:
            normalizeddata = 2       
        print("female", creatinine, normalizeddata)

    else:  # Male
        if creatinine < 0.65:
            normalizeddata = 0
        elif creatinine <= 1.07:
            normalizeddata = 1
        else:
            normalizeddata = 2
        print("male", creatinine, normalizeddata)
    
    after_normalization.iloc[i, ] = normalizeddata

normalized_allcoviddata_pandas["Creatinine"] = after_normalization
    

before_normalization = allcoviddata_pandas["Lactate_dehydrogenase"]
def Lactate_dehydrogenase_func(x):
    if x < 220:
        return 0
    elif x <= 440:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(Lactate_dehydrogenase_func)
normalized_allcoviddata_pandas["Lactate_dehydrogenase"] = after_normalization


before_normalization = allcoviddata_pandas["C_reactive_protein"]
def C_reactive_protein_func(x):
    if x < 0.14:
        return 0
    elif x <= 10:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(C_reactive_protein_func)
normalized_allcoviddata_pandas["C_reactive_protein"] = after_normalization


before_normalization = allcoviddata_pandas["D_dimer"]
def D_dimer_func(x):
    if x < 1:
        return 0
    elif x <= 2:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(D_dimer_func)
normalized_allcoviddata_pandas["D_dimer"] = after_normalization


before_normalization = allcoviddata_pandas["Body_temperature"]
def Body_temperaturet_func(x):
    if x < 36:
        return 0
    elif x < 37.5:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(Body_temperaturet_func)
normalized_allcoviddata_pandas["Body_temperature"] = after_normalization


before_normalization = allcoviddata_pandas["Time_from_onset"]
def Time_from_onset_func(x):
    if x <= 7:
        return 0
    elif x <= 14:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(Time_from_onset_func)
normalized_allcoviddata_pandas["Time_from_onset"] = after_normalization

before_normalization = allcoviddata_pandas["Age"]
def Age_func(x):
    if x < 60:
        return 0
    elif x < 80:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(Age_func)
normalized_allcoviddata_pandas["Age"] = after_normalization


before_normalization = allcoviddata_pandas["BMI"]
def BMI_func(x):
    if x < 18.5:
        return 0
    elif x <= 24.9:
        return 1
    else:
        return 2
after_normalization = before_normalization.map(BMI_func)
normalized_allcoviddata_pandas["BMI"] = after_normalization

normalized_allcoviddata_pandas.to_excel(FileNameOfNormalizedPatientsData, sheet_name='Sheet1')

'''
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.where.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mask.html
# tfo1 = tfo.mask(tfo <= 7, other = 0)
# print(tfo1)
# tfo2 = tfo.mask(((7 < tfo) & (tfo <= 14)), other = 1)
# print(tfo2)
# tfo3 = tfo1.where(tfo > 14, other = 2)
# print(tfo3)

# Replace "Severity" item: mild -> 0, moderateI -> 1, moderateII -> 2, severity -> 3
#   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
#allcoviddata_pandas = allcoviddata_pandas.replace({'Severeity':{"mild":0, "moderateI":1, "moderateII":2, "severe":3}})
# print(allcoviddata_pandas)
'''
