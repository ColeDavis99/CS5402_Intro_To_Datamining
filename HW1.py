import pandas as pd
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import spearmanr
import numpy as np
import seaborn as sea
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Read in raw census data
df = pd.read_csv('census.csv')

#Convert the data type of this column (Done for task 4)
df["population-wgt"] = df["population-wgt"].astype('float64')


'''
#####################################
Task 1) Make date format consistent
#####################################
'''
#Iterate over the date column of the dataframe
date_split = []

for rownum, data in df.iterrows():
    date_split = data["date"].split("/")            # Split each date into (ideally) 3 sections via "/" splicing: MM/DD/YYYY

    if(len(date_split) != 3):                           # 23 entries in the dataset did not split into three parts via "/" splicing. 21 of the 23 were representing May 7th, so I made the 
        df.at[rownum, "date"] = "5/07/1994"             # decision to replace them all with May 7th due to the small amount of inaccuracy this produced relative to the size of the dataset.
    else:
        if(int(date_split[0]) not in range(1,13)):          # The three if statements nested within this else statement ensure that the month and date values are (mostly) valid.      
            print("Month out of range " + data["date"])     # All month values were found to be within the range [1,12] and day values between [1,31]. Additional checks were NOT made to 
        if(int(date_split[1]) not in range(1,32)):          # ensure imaginary dates like Feb. 30th were not present. 
            print("Day out of range " + data["date"])
        if(date_split[2] != '1994'):                        # Several "non-1994" values for the year were identified and changed to "1994"
            #print("Year out of range " + data["date"])
            df.at[rownum, "date"] = date_split[0] + "/" + date_split[1] + "/1994"


'''
##########################################
Task 2) Discretize Age Column into 10 bins
##########################################
'''
discretizer = EqualWidthDiscretiser(bins=10, variables=["age"])     # Use feature_engine library to discretize this column.
df = discretizer.fit_transform(df)


'''
##################################################
Task 3) Replace workclass "?" values with "Other"
Task 4) Normalize values within Population-wgt column
Task 5) Replace Occupation "?" values with "Other"
Task 6) Make sex column "Male" or "Female"
Task 8) Replace Native-Country "?" values with "Unspecified"
##################################################
'''
#Iterate over dataframe again and apply some changes.
for rownum, data in df.iterrows():
    # Task 3
    if("?" in df.at[rownum, "workclass"]):
        df.at[rownum, "workclass"] = "Other"

    # Task 4
    # Normalized Value = [cell value - min(column)] / [max(column) - min(column)]
    normalized_val = (df.at[rownum, "population-wgt"] - df["population-wgt"].min()) / (df["population-wgt"].max() - df["population-wgt"].min())
    df.at[rownum, "population-wgt"] = normalized_val

    # Task 5
    if("?" in df.at[rownum, "occupation"]):
        df.at[rownum, "occupation"] = "Other"

    # Task 6
    # Distinct values were found to be: [' Male' ' Female' 'm' 'M' 'F' 'f' 'male' 'female' 'fem']
    if(df.at[rownum, "sex"] in [" Male", "m", "M", "male"]):
        df.at[rownum, "sex"] = "Male"
    else:
        df.at[rownum, "sex"] = "Female"

    #Task 8
    if("?" in df.at[rownum, "native-country"]):
        df.at[rownum, "native-country"] = "Unspecified"


'''
#######################################################
Task 7) Discretize hours-per-week with equal frequency 
#######################################################
'''
discretizer = EqualFrequencyDiscretiser(q=30, variables = ['hours-per-week'])
df = discretizer.fit_transform(df)


'''
#############################################################################
Task 9) Perform Chi-Squared test for independence between 10 nominal columns.  
#############################################################################
'''
SIGNIFICANCE = 0.05  #0.05 original
P = 1 - SIGNIFICANCE

critical_value = 0.0
nominal_colnames = ["age", "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "hours-per-week", "native-country"]
occurrence_dict = {}    #Key = (distinctCol_A_Val, distinctCol_B_Val)   Value = # of times this pair of values are found together in the same row
contingency_table = []  #This will be a 2D list

'''Loop over each pair of nominal columns'''
for i in range(len(nominal_colnames)):          
    for q in range(i+1, len(nominal_colnames)):        
        '''Creating contingency table for this particular pair of nominal columns'''
        #Get listing of each column's distinct values
        distinct_list_1 = df[nominal_colnames[i]].unique()
        distinct_list_2 = df[nominal_colnames[q]].unique()

        #Loop over each pair of distinct values and initialize the dictionary that will store how many times they appear together in the same row
        for x in range(len(distinct_list_1)):
            for y in range(len(distinct_list_2)):
                occurrence_dict[(distinct_list_1[x], distinct_list_2[y])] = 0

        #Get a count of how many times each distinct pair of values are found together. (The sum of the values of this dictionary should equal the # of rows in the original dataset)
        for rownum, data in df.iterrows():
            occurrence_dict[(df.at[rownum, nominal_colnames[i]], df.at[rownum, nominal_colnames[q]])] += 1

        #Initialize a contingency table, a list of lists
        contingency_table = [[] for a in range(len(distinct_list_1))]

        #Convert the dictionary "occurrence_dict" into a contingency table. (Pandas wants it formatted a specific way).
        for k in range(len(distinct_list_1)):
            for d in range(len(distinct_list_2)):
                contingency_table[k].append(occurrence_dict[(distinct_list_1[k], distinct_list_2[d])])

        #Create the Pandas version of the contingency table
        contingency_df = pd.DataFrame(contingency_table, index=distinct_list_1, columns=distinct_list_2)

        '''Compute the chi^2 value for this column pair now that the contingency table is created'''
        chi, pval, dof, exp = chi2_contingency(contingency_df)
        print(nominal_colnames[i] + " and " + nominal_colnames[q])
        critical_value = chi2.ppf(P, dof)

        print('chi=%.6f, critical value=%.6f' % (chi, critical_value))
        if(chi > critical_value):
            print("Dependent")
        else:
            print("Independent")

        
        #Erase keys and values now that we're moving on to the next column pair. Also delete the 2D array contingency table and Pandas contingency table now that we have the chi^2 value.
        occurrence_dict.clear()
        del contingency_table
        del contingency_df

        print()


'''
#################################################################################
Task 10) Perform Spearman test for independence between the non-nominal columns.  NOTE: THEY ARE ALL INDEPENDENT.
#################################################################################
'''
non_nominal_idxs = [0, 3, 5, 11, 12]

for i in range(len(non_nominal_idxs)):
    for q in range(i+1, len(non_nominal_idxs)):
        X = df.iloc[:, non_nominal_idxs[i]].values.reshape(-1, 1)
        Y = df.iloc[:, non_nominal_idxs[q]].values.reshape(-1, 1)
        corr, p_value = spearmanr(X, Y)


        print(df.columns.values[non_nominal_idxs[i]], df.columns.values[non_nominal_idxs[q]])
        if(abs(corr) < 0.8):
            print("Independent")
        else:
            print("Dependent")
        print()


'''
#################################################################################
Task 11) Change all non-numeric columns to numeric and standardize each attribute
#################################################################################
'''
non_numeric_cols = list(df.columns.values)
non_numeric_cols.pop()  #Remove the decision attribute from the list to normalize and standardize

distinctVals = []
replace_dict = {}

#Loop over all cols in the dataframe
for i in range(len(non_numeric_cols)):
    distinctVals = df[non_numeric_cols[i]].unique()
    
    #Create a dictionary of each distinct values in the column and assign it the number it maps to
    for q in range(len(distinctVals)):
        replace_dict[distinctVals[q]] = q
        
    #Replace the values in the dataframe with the numeric mapping we just computed.
    for rownum, data in df.iterrows():
        df.at[rownum, non_numeric_cols[i]] = replace_dict[df.at[rownum, non_numeric_cols[i]]]

    replace_dict.clear()
    del distinctVals

#Standardize the numeric columns we just mapped
df[non_numeric_cols] = StandardScaler().fit_transform(df[non_numeric_cols])


'''
#################################################################################
Task 12) Perform PCA 
#################################################################################
'''


#Write out to a different csv
df.to_csv('clean_census.csv', index=False)# index=False so no index column