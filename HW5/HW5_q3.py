import numpy as np
import pandas as pd
from sklearn.svm import SVC
from pomegranate import *

# Read in the datasets to create the Bayesian Networks
df = pd.read_csv('contact-lenses (copy).csv')
#df2 = pd.read_csv('hypothyroid (copy).csv')


'''
For each column in the dataset, calculate the ratio of each distinct vals occurrance, grouped 
by the distinct value of the decision attribute. These are the raw conditional probabilities
which will soon be altered and laplace'd.
'''

#This is the col name of the decision attribute
decAttr = list(df)[-1]

#The division operation should happen after all rows are iterated over and # of occurrences is computed.
conditional_prob_table = dict() 

#Make a dictionary with a concatenated key of column name, unique non-decision attribute, and unique decision attribute
colList = []
distinctNonDecValList = []
distinctDecValList = []

for col in list(df[:-1]):                               #Loop over all but the last column of the dataframe
    for distinctNonDecVal in df[col].unique():          #Loop over each unique non-decision attribute value
        for distinctDecVal in df[decAttr].unique():     #Loop over each unique decision attribute value
            conditional_prob_table[col+distinctNonDecVal+distinctDecVal] = 0

            #Store these, we'll need them to iterate over the conditional probability list
            if col not in colList:
                colList.append(col)
            if distinctDecVal not in distinctDecValList:
                distinctDecValList.append(distinctDecVal)
            if distinctNonDecVal not in distinctNonDecValList:
                distinctNonDecValList.append(distinctNonDecVal)


#Increment each combination in the dictionary. The next step is to divide each value by the total # of rows in the dataset. That # is the conditional probability.
colctr = 0
dflen = len(colList)
for idx, row in df.iterrows():
    colctr = 0
    for cell in row:
        print(row.index[colctr])            #colname
        print(cell)                         #cell value
        print(row[dflen-1])                 #Decision attribute value for the row
        print()

        conditional_prob_table[str(row.index[colctr]) + str(cell) + str(row[dflen-1])] += 1
        colctr += 1
    print()


#Now calculate the conditional probability for each combination.
for col in list(df[:-1]):                               #Loop over all but the last column of the dataframe
    for distinctNonDecVal in df[col].unique():          #Loop over each unique non-decision attribute value
        for distinctDecVal in df[decAttr].unique():     #Loop over each unique decision attribute value
            conditional_prob_table[col+distinctNonDecVal+distinctDecVal] = conditional_prob_table[col+distinctNonDecVal+distinctDecVal] / len(df)


#Debugging Loop
for key in conditional_prob_table:
    print("Key: " + key + " " + "Value: " + str(conditional_prob_table[key]))


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
    print(key)
    labelctr = 0
    for val in value_counts[key]:
        conditional_probs[key][value_counts[key].axes[0][labelctr]] = val
        print(value_counts[key].axes[0][labelctr])
        labelctr += 1
    print()


#Calculate the conditional probability for each distinct value
total_rows = len(df1)
for key1 in conditional_probs:
    print("Col: " + key1)
    for key2 in conditional_probs[key1]:
        conditional_probs[key1][key2] = conditional_probs[key1][key2] / total_rows
        print("Distinct Val: " + key2 + " " + str(conditional_probs[key1][key2]))
    print()
'''