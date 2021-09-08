import pandas as pd
from feature_engine.discretisation import EqualWidthDiscretiser

#Read in raw census data
df = pd.read_csv('census.csv')


#Iterate over the date column of the dataframe
date_split = []
date_temp = ""
date_problem = False

for rownum, data in df.iterrows():
    date_split = data["date"].split("/")            # Split this date into (ideally) 3 sections. MM/DD/YYYY

    if(len(date_split) != 3):                       # 23 entries in the dataset did not split into three parts via "/" splicing. 21 of the 23 were representing May 7th, so I made the 
        df.at[rownum, "date"] = "5/07/1994"          # decision to replace them all with May 7th due to the large size of the dataset relative to the amount of inaccuracy this produced.
    else:
        if(int(date_split[0]) not in range(1,13)):          # The three if statements nested within this else statement ensure that the month and date values are (mostly) valid.      
            print("Month out of range " + data["date"])     # All month values were found to be within the range [1,12] and day values between [1,31]. Additional checks were NOT made to 
        if(int(date_split[1]) not in range(1,32)):          # ensure imaginary dates like Feb. 30th were not present. 
            print("Day out of range " + data["date"])
        if(date_split[2] != '1994'):                        # Several "non-1994" values for the year were identified and changed to "1994"
            print("Year out of range " + data["date"])
            df.at[rownum, "date"] = date_split[0] + "/" + date_split[1] + "/1994"