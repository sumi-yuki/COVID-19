#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:32:15 2023

@author: Yuki Sumi

"""

# Loading the library

# pandas https://pandas.pydata.org
# To install,
# pip install pandas
import pandas as pd

# openpyxl https://pypi.org/project/openpyxl/
# To install,
# pip install openpyxl==3.1.2

# UMAP https://umap-learn.readthedocs.io/en/latest/
# To install,
# pip install umap-learn==0.5.6
import umap

# Matplotlib https://matplotlib.org/3.5.3/index.html
# pip install matplotlib
import matplotlib.pyplot as plt

# file name of covid-19 patients' data
#FileNameOfPatientsData = "covid_data.xlsx"
FileNameOfPatientsData = "covid_data_scored.xlsx"

# List of Items used to analyse
InputItemList = ["Severity","Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph"]
# Using all data
AnalyzeItemList = ["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph"]

# Read the covid-19 patients Excel data file
allcoviddata_pandas =pd.read_excel(FileNameOfPatientsData)
# Extract data to analyse
coviddata_pandas = allcoviddata_pandas[AnalyzeItemList]
print("coviddata_pandas", coviddata_pandas)

# Extracting severity data
severity = allcoviddata_pandas["Severity"]
print("severity", severity)

def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
#def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    # https://umap-learn.readthedocs.io/en/latest/api.html#module-umap.umap_
    reducer = umap.UMAP(
        n_neighbors = n_neighbors,
        min_dist = min_dist,
        n_components = n_components,
        low_memory = False, 
    )
    u = reducer.fit_transform(coviddata_pandas);

    fig = plt.figure(figsize = (12,12))
    if n_components == 1:
        scatter = plt.scatter(u[:,0], u[:,0], c=severity, cmap="coolwarm", alpha=0.1)
    elif n_components == 2:
        scatter = plt.scatter(u[:,0], u[:,1], c=severity, cmap="coolwarm")
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(u[:,0], u[:,1], u[:,2], c=severity, cmap="coolwarm", s=100)
    plt.legend(*scatter.legend_elements())
    figure_title = 'UMAP {}'.format(n_components) + 'D,  metrics="' + metric + '",  n_neighbors = {}'.format(n_neighbors) + ',  min_dist = {}'.format(min_dist)
    plt.title(figure_title, fontsize=20)
    save_file_name = 'umap_n{}'.format(n_components) + 'Dmetrics_' + metric + '_umap_n_neighbors{}'.format(n_neighbors) + '_min_dist{}'.format(min_dist) + ".jpg"
    plt.savefig(save_file_name, dpi=2400)
    plt.show()

# Changing the parameters
for neighbor in (10, 100, 200): # n_neighbors
#for n in (2, 5, 10, 20, 50, 100, 200): # n_neighbors
    for distance in (0.1, 0.5, 0.99): # min_dist
    #for d in (0.0, 0.1, 0.25, 0.5, 0.8, 0.99): # min_dist
        for m in ("euclidean", "canberra", "mahalanobis"): # metric
            for dimension in (2, 3):
                draw_umap(n_neighbors=neighbor, min_dist=distance, metric=m, n_components=dimension)
