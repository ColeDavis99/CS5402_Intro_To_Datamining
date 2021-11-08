import numpy as np
import pandas as pd
import copy
from sklearn.svm import SVC
from pomegranate import *



def CreateBayesianNetwork(df):
        
    '''
    For each col in the dataset, calculate the ratio of each distinct vals occurrance, grouped 
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

    for col in list(df)[:-1]:                               #Loop over all but the last column of the dataframe
        for distinctNonDecVal in df[col].unique():          #Loop over each unique non-decision attribute value
            for distinctDecVal in df[decAttr].unique():     #Loop over each unique decision attribute value
                conditional_prob_table[col+distinctNonDecVal+distinctDecVal] = 0

                #Store these, we'll need them to iterate over the conditional probability list and know which node we're talking about
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
            if(row.index[colctr] != decAttr):
                conditional_prob_table[str(row.index[colctr]) + str(cell) + str(row[dflen])] += 1
                colctr += 1


    #Store a copy of the count of each combination of colname, cell value, and decision attribute.
    combination_snapshot = copy.deepcopy(conditional_prob_table)

    colctr = 0
    #Now calculate the conditional probability for each non-dec combination.
    for col in list(df)[:-1]:                               #Loop over all but the decision attr column of the dataframe
        for distinctNonDecVal in df[col].unique():          #Loop over each unique non-decision attribute value
            for distinctDecVal in df[decAttr].unique():     #Loop over each unique decision attribute value                                   # num of times a col has this non-dec val occur
                conditional_prob_table[col+distinctNonDecVal+distinctDecVal] = conditional_prob_table[col+distinctNonDecVal+distinctDecVal] / df[col].value_counts().at[distinctNonDecVal]
            colctr += 1

    #Now calculate the conditional probability for each dec combination.
    for val in distinctDecValList:
        conditional_prob_table[decAttr+val] = df[decAttr].value_counts().at[val]/len(df)




    '''##########################################################################################
    #Apply Laplace smoothing to all values within the conditional_prob_table
    ##########################################################################################'''

    #Laplace Loop for dec attributes
    LMBDA = 1
    for val in distinctDecValList:
        conditional_prob_table[decAttr+val] = (df[decAttr].value_counts().at[val] + LMBDA) / (len(df) + (len(distinctDecValList) * LMBDA))
        # print(conditional_prob_table[decAttr+val])


    #Laplace Loop for non-dec attributes
    for col in list(df)[:-1]:                              
        for distinctNonDecVal in df[col].unique():          
            for distinctDecVal in df[decAttr].unique():
                conditional_prob_table[col+distinctNonDecVal+distinctDecVal] = (combination_snapshot[col+distinctNonDecVal+distinctDecVal] + LMBDA) / ((df[decAttr].value_counts().at[distinctDecVal]) + (len(list(df[col].unique())) * LMBDA))

    #Debugging Loop 2
    '''
    for key in conditional_prob_table:
        print("Key: " + key + ": \t" + str(conditional_prob_table[key]))
    '''



    '''#######################################
    Create the Bayesian Network w/Pomegranate 
    ##########################################'''
    #Create Decision Attribute dictionary
    decAttrDict = dict()
    for val in distinctDecValList:
        decAttrDict[val] = conditional_prob_table[decAttr+val]

    decisionTable = DiscreteDistribution(decAttrDict)


    #Create a list of Conditional Probability Tables
    conditionalProbTableList = list()

    outerTempList = list()
    innerTemplist = list()
    for col in list(df)[:-1]:                           #For each non-dec column
        for distinctNonDecVal in df[col].unique():      #For each distinct value in that column
            for distinctDecVal in df[decAttr].unique():
                innerTemplist.append(str(distinctDecVal))
                innerTemplist.append(str(distinctNonDecVal))
                innerTemplist.append(conditional_prob_table[col+distinctNonDecVal+distinctDecVal])


                outerTempList.append(copy.deepcopy(innerTemplist))
                innerTemplist.clear()

        conditionalProbTableList.append(ConditionalProbabilityTable(outerTempList, [decisionTable]))
        outerTempList.clear()


    # print(conditionalProbTableList)



    #Create decision attribute node object
    decNode = Node(decisionTable, name=decAttr)

    #Create a list of node objects that will make up the Bayesian network (not the decision attribute node yet though)
    nodeList = list()
    nodeNameCtr = 0
    for table in conditionalProbTableList:
        nodeList.append(Node(table, name=colList[nodeNameCtr]))
        nodeNameCtr += 1


    #Create the model and bake it
    model = BayesianNetwork("HW5_Network")
    model.add_state(decNode)

    print("========= Conditional Probability Tables ============")
    #Add the nodes
    for node in nodeList:
        print("NODE: " + node.name)
        print(node.distribution)
        print()
        print()
        model.add_state(node)

    #Add the edges
    for node in nodeList:
        model.add_edge(decNode, node)


    #Like an oven
    model.bake()
    return model



'''
###########
  DRIVER
###########
'''
# Read in the datasets to create the Bayesian Networks
df = pd.read_csv('contact-lenses (copy).csv')
# df = pd.read_csv('hypothyroid (copy).csv')

#Create a Bayesian Network from the dataframe
model = CreateBayesianNetwork(df)


'''
################
  PREDICTIONS
################
'''
'''
#Contact-lenses.csv hardcode prediction
print(model.predict([['presbyopic', None, None, None, 'soft']]))

#Hypothyroid.csv hardcode prediction
print(model.predict([['U', None, None, None, None, None, None, None, None, 'negative']]))
'''