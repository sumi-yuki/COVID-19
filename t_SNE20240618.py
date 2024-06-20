#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:31:16 2023

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

# numpy https://numpy.org/ja/
# To install,
# pip install numpy
# import numpy as np

# scikit-learn https://scikit-learn.org/stable/index.html
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE
from sklearn.manifold import TSNE

# Matplotlib https://matplotlib.org/3.5.3/index.html
# pip install matplotlib
import matplotlib.pyplot as plt

# file name of covid-19 patients' data
# Loading data file name
#FileNameOfPatientsData = "covid_data_scored.xlsx"
FileNameOfPatientsData = "covid_data.xlsx"

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

# t-SNE
def draw_tsne(perplexity=30.0, early_exaggeration=12.0, init='pca', n_components=2):
    reducer = TSNE(perplexity=perplexity, early_exaggeration=early_exaggeration, init=init, n_components=n_components, method='exact')
    u = reducer.fit_transform(coviddata_pandas)

    fig = plt.figure(figsize = (12,12))
    if n_components == 1:
        scatter = plt.scatter(u[:,0], u[:,0], c=severity, cmap="coolwarm", alpha=0.1)
    elif n_components == 2:
        scatter = plt.scatter(u[:,0], u[:,1], c=severity, cmap="coolwarm")
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(u[:,0], u[:,1], u[:,2], c=severity, cmap="coolwarm", s=100)
    plt.legend(*scatter.legend_elements())
    title='t-SNE {0}'.format(n_components) + 'D, perplexity{0}'.format(perplexity) + ', early_exaggeration {0}'.format(early_exaggeration) + ', init = {0}'.format(init)
    plt.title(title)
    save_file_name = 't-SNE{0}'.format(n_components) + 'Dperplexity{0}'.format(perplexity) + 'early_exaggeration{0}'.format(early_exaggeration) + 'init={0}'.format(init) + ".jpg"
    plt.savefig(save_file_name, dpi=2400)
    plt.show()

# Changing the parameters
# Perplexity
for perplexity in (10, 30, 50):
    # early_exaggeration
    for early_exaggeration in (6, 12, 24):
        # init
        for init in ["random", "pca"]:
            for dimension in (2, 3):
                draw_tsne(perplexity=perplexity, early_exaggeration=early_exaggeration, init=init, n_components=dimension)