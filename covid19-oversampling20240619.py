# -*- coding: utf-8 -*-
"""
Created by Yuki Sumi on 2/June/2024

"""

# installing modules on python 3.9.19

# numpy https://numpy.org
# pip install numpy==1.26.4

# pandas https://pandas.pydata.org
# pip install pandas==2.2.2

# openpyxl https://pypi.org/project/openpyxl/
# To install,
# pip install openpyxl==3.1.2

# tensorflow https://www.tensorflow.org/?hl=en
# pip install tensorflow==2.11.1

# tensorflow.js https://www.tensorflow.org/js?hl=en
# pip install tensorflowjs==4.4.0

# Matplotlib https://matplotlib.org
# To install,
# pip install matplotlib==3.9.0

# scikit-learn https://scikit-learn.org/stable/
# pip install scikit-learn==1.4.2

# imbalanced-learn https://imbalanced-learn.org/stable/index.html
# pip install imbalanced-learn==0.12.2

# pip install jax==0.4.21
# pip install jaxlib==0.4.21

# Loading the library
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout

import pandas as pd


# To prevent OMP Abort error with anaconda
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Declaring variables for 4 categories
number_of_classes = 4
# Declaring variables for 2 categories
# number_of_classes = 2

# file name of covid-19 patients' data
FileNameOfPatientsData = "covid_data_scored.xlsx"
#FileNameOfPatientsData = "covid_data.xlsx"

# List of Items used to analyse
InputItemList = ["Severity","Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph"]
# Using all data
TrainingItemList = ["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma","Chest_radiograph"]
# Using clinical data
# TrainingItemList = ["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma"]
# Using clinical-blood data
# TrainingItemList = ["Time_from_onset","Age","Gender","BMI","SpO2","Body_temperature","D_dimer","C_reactive_protein","Lactate_dehydrogenase","Lymphocyte","Creatinine","Urea_nitrogen","Diabetes","Hypertension","Hyperlipemia","Hyperuricemia","Chronic_obstructive_pulmonary_disease","Cardio_vasucular_disease","Smoking_history","Malignant_neoplasm","Asthma"]

# Read the covid-19 patients Excel data file
allcoviddata_pandas =pd.read_excel(FileNameOfPatientsData)

# Replace "Severity" item: mild -> 0, moderateI -> 1, moderateII -> 2, severity -> 3
#   https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html
# allcoviddata_pandas = allcoviddata_pandas.replace({'Severeity':{"mild":0, "moderateI":1, "moderateII":2, "severe":3}})
# print(allcoviddata_pandas)

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

# Correct for data imbalance by oversampling using SMOTE
from imblearn.over_sampling import SMOTEN #SMOTE, KMeansSMOTE, SVMSMOTE, ADASYN, SMOTENC, SMOTEN, BorderlineSMOTE, RandomOverSampler
sm = SMOTEN() #you can select other oversampling methods like ADASYN, KMeansSMOTE()
#from imblearn.over_sampling import SMOTE #, KMeansSMOTE, SVMSMOTE, ADASYN
#sm = SMOTE() #you can select other oversampling methods like ADASYN, KMeansSMOTE()
# for downsampling, use following 2 instructions instead
#  from imblearn.under_sampling import RandomUnderSampler
#  sm = RandomUnderSampler()
x_resampled, y_resampled = sm.fit_resample(data_train_numpy, labels_train_numpy)
print(x_resampled.shape[1])
#print(y_resampled.shape)
print(y_resampled[..., np.newaxis].shape)
covid_xtrain = x_resampled
covid_ytrain = y_resampled[..., np.newaxis]

covid_ytest = coviddata_test_pandas['Severity'].values
covid_xtest = coviddata_test_pandas[TrainingItemList].values

number_of_ExplanatoryVariables = covid_xtrain.shape[1]

# Define neural networks
model = Sequential()
model.add(Dense(128, input_shape=(number_of_ExplanatoryVariables,)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))
model.add(Dense(units=128))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.4))
model.add(Dense(units=128))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
#model.add(Dense(units=32))
#model.add(BatchNormalization())
#model.add(LeakyReLU(alpha=0.3))
#model.add(Dropout(0.1))

# number of categories output
model.add(Dense(number_of_classes,  activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='Adam' ,
              metrics=["accuracy", "Recall", "Precision"]
              )

# View the shape of the created neural circuit
model.summary()

# One-hot encode
from tensorflow.keras.utils import to_categorical
covid_ytest_categorical = to_categorical(covid_ytest)
covid_ytrain_categorical = to_categorical(covid_ytrain)

history = model.fit(covid_xtrain, covid_ytrain_categorical, 
                    batch_size=32, epochs=500, 
                    validation_split=0.1)

# View learning progress in a graph
import matplotlib.pyplot as plt

history_dict = history.history
# Put Loss (error from correct answer) into the loss_values
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
# Put accuracy in ACC
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
# Make a list from 1 to epoch number
epochlist = range(1, len(loss_values) +1)

# Create an accuracy graph
plt.plot(epochlist, acc, 'go', label='Accuracy at training')
plt.plot(epochlist, val_acc, 'b', label='Accuracy at validation')
# Create a graph of Loss (error from correct answer)
plt.plot(epochlist, loss_values, 'mo', label='Loss at training')
plt.plot(epochlist, val_loss_values, 'r', label='Loss at validation')

# Title
plt.title('Epoch - accuracy and loss')
plt.xlabel('epoch')
plt.legend()
# View and save the graph
plt.savefig("Training_all_SMOTEN6.jpg", dpi=2400)
plt.show()

# Use test datasets to display loss and percentage of correct answers
score = model.evaluate(covid_xtest, covid_ytest_categorical)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Test Recall:",score[2])
print("Test Precision:",score[3])

prediction = model.predict(covid_xtest, verbose=0)
print(covid_ytest)
y_pred = prediction.argmax(axis=1)
print(y_pred)

# Create HDF file
model.save('all_SMOTEN6.keras')

# # Convert python keras file to js file
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "./tfjs_model_all_SMOTEN6")

print("ground truth 4 classes", covid_ytest)
prediction = model.predict(covid_xtest, verbose=0)
y_pred = prediction.argmax(axis=1)
print("prediction 4 classes", y_pred)

covid_ytest = np.where(covid_ytest < 2, 0, 1) 
print("ground truth 2 classes", covid_ytest)
y_pred = np.where(y_pred < 2, 0, 1) 
print("prediction 2 classes", y_pred)
print("accuracy 2 classes", np.sum(covid_ytest == y_pred) / number_of_test)




