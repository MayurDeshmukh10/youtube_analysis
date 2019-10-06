import numpy as np 
import pandas as pd 
import seaborn as sns
from textblob import TextBlob


data = pd.read_csv("train.csv",error_bad_lines = False)

data.head()

data.shape

data.nunique()

data.info()

pol = []

for i in data["Comment"].values:
        try:
            analysis = TextBlob(i)
            result = analysis.sentiment.polarity

            if result == 0:
                pol.append(0)
            else:
                pol.append(1)

        except:
            pol.append(0)


dict = {"Video ID": data["Video ID"],"Title": data["Title"],"Comment" : data["Comment"],"Polarity" : pol}


df = pd.DataFrame(dict)

df.to_csv("updated_pol.csv",header=False,index=False)




