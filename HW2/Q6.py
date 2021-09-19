from sklearn import tree
import pandas as pd
import numpy
import graphviz

df = pd.read_csv('hw2_prob6_Copy.csv')
r,c = df.shape


#Replace nominal attributes with numeric for doing the CART algorithm.
df = df.replace({'outlook': r'good'}, {'outlook':1}, regex=True)
df = df.replace({'outlook': r'bad'}, {'outlook':0}, regex=True)

df = df.replace({'temperature': r'warm'}, {'temperature':1}, regex=True)
df = df.replace({'temperature': r'cool'}, {'temperature':0}, regex=True)

df = df.replace({'humidity': r'high'}, {'humidity':1}, regex=True)
df = df.replace({'humidity': r'normal'}, {'humidity':0}, regex=True)

df = df.replace({'windy': r'TRUE'}, {'windy':1}, regex=True)
df = df.replace({'windy': r'FALSE'}, {'windy':0}, regex=True)



X = df.iloc[:, 0:c-1].values    # non-decision attributes
y = df.iloc[:, c-1].values      # decision attribute
clf = tree.DecisionTreeClassifier(criterion="gini")
clf = clf.fit(X,y)

attrNames = list(df.columns)
classNames = list(set(df["play"].values))
classNames.sort()
classNames = numpy.array(classNames)
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=attrNames[0:c-1],
    class_names=classNames,filled=True,
    rounded=True,
    special_characters=True)

graph = graphviz.Source(dot_data)

graph.render("Trading_Decision_Tree") # see Trading_Decision_Tree.pd
