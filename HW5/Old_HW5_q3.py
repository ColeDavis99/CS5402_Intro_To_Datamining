import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Read in the datasets to create the Bayesian Networks
df1 = pd.read_csv('contact-lenses (copy).csv')
#df2 = pd.read_csv('hypothyroid (copy).csv')


'''
For each column in the dataset, calculate the ratio of each distinct vals occurrance compared
to the total # of rows in the dataset. These are the raw conditional probability which will soon
be laplace'd.
'''
#Generate a Pandas "Series" object that contains the count of each distinct value per column.
value_counts = {}
for col in df1:
    value_counts[col] = df1[col].value_counts()

#Loop over the Series object and extract the values we want. Store in a dict of dicts.
conditional_probs = dict()
labelctr = 0
for key in value_counts:
    conditional_probs[key] = dict()
    # print(key)
    labelctr = 0
    for val in value_counts[key]:
        conditional_probs[key][value_counts[key].axes[0][labelctr]] = val
        # print(value_counts[key].axes[0][labelctr])
        labelctr += 1
    # print()

#Calculate the conditional probability for each distinct value
total_rows = len(df1)
for key1 in conditional_probs:
    print("Col: " + key1)
    for key2 in conditional_probs[key1]:
        conditional_probs[key1][key2] = conditional_probs[key1][key2] / total_rows
        print("Distinct Val: " + key2 + " " + str(conditional_probs[key1][key2]))
    print()