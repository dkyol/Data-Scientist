Deep Learning Model Categorical Features - PCA transform

I analyzed demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. Then I used unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. 

Then I applied the model on a third dataset with demographics information for targets of a marketing campaign for the company to predict which individuals are most likely to convert into becoming customers for the company. 

The results of the analysis was a model could be used to improve the business in the current form.

The approach here was use all of the data available in it raw form before completing feature scaling, feature creation and/or other pre-processing steps. These results were accomplished without incorporating more than 200 features that may require additional domain knowledge to incorporate effectively.

Experimentation with different model architectures, different out of the box techniques, hyperparameters could also improve the accuracy/recall/precision of the model. 

A high-level description of the code and results can be found here:
https://medium.com/@dkylemiller/machine-learning-with-categorical-variables-4f7d1427004a

Datasets:

Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).

Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).

Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).

Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

I used the following libraries: 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, ward, single

from sklearn.model_selection import train_test_split 

from keras import metrics

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score
