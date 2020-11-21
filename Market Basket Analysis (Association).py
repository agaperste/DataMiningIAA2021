import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml
import os


# load the raw csv file
os.chdir("/Users/zhanyina/Documents/MSA/AA502 Analytics Methods and Applications I/Data Mining/Asgnmts/1")
df_raw = pd.read_csv("orderData.csv")
df_raw.head(20)

# create a transaction column to identify a unique transaction
df_raw['transactionId'] = df_raw['orderNo'].astype(str) + df_raw['seatNo'].astype(str)
df_raw.head(20)

# visualize the most populater items ordered
sns.countplot(x = 'item', data = df_raw, order = df_raw['item'].value_counts().iloc[:10].index)
plt.xticks(rotation=90)

''' finding the most popular meal --> Filet Mignon/Blackstone Merlot/Seasonal Veg'''
df_hold = df_raw.groupby('transactionId').agg('/'.join)
df_freq = pd.crosstab(index=df_hold["item"], columns="count").sort_values('count',ascending = False) 
# percent of the most popular 3-item meal in the dataset --> 1.16%
881/df_raw.transactionId.nunique()

'''======= BEGIN market basket analysis on this data set ======='''

df = df_raw.groupby(['transactionId','item']).size().reset_index(name='count')
basket = (df.groupby(['transactionId', 'item'])['count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('transactionId'))

#The encoding function
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
basket_sets = basket.applymap(encode_units)

# we first tried with min_support 0.01, but couldn't get all meat items paired up
# hence tried 0.005 again to get all meat paired
# we used support as cut-off but looked at confidence to pick
# max_len=2 --> limiting antecedent and consequence to be a single item
frequent_itemsets = apriori(basket_sets, min_support=0.005, use_colnames=True, max_len=2)
rules = association_rules(frequent_itemsets, metric="lift")

# sort by confidence descending
rules.sort_values('confidence', ascending = False, inplace = True)

# display and export
pd.set_option("display.max_rows", None, "display.max_columns", None)
rules.head(20)

# get a list of unique items in this restaurant
print(df_raw.item.unique())
all_meat = ["Salmon", "Pork Chop", "Sea Bass", "Duck Breast", "Swordfish", "Pork Tenderloin", 
            "Filet Mignon", "Roast Chicken"]

all_wine = ["Oyster Bay Sauvignon Blanc", "Three Rivers Red", "Total Recall Chardonnay", 
            "Innocent Bystander Sauvignon Blanc", "Duckhorn Chardonnay", "Louis Rouge",
            "Helben Blanc", "Single Vineyard Malbec", "Adelsheim Pinot Noir", 
            "Cantina Pinot Bianco", "Brancott Pinot Grigio", 
            "Blackstone Merlot", "Echeverria Gran Syrah"]

# filtering antecedents to only include meat and consequence to only include wine
meat_antecedent_rules = []

for meat in all_meat:
    meat_antecedent_rules.append(rules[ (rules['antecedents'].astype(str).str.contains(meat))])

# checking my list of 8 meat's rules
meat_antecedent_rules
len(meat_antecedent_rules)

# flatten a list of dataframes into 1
flat_meat_rules = pd.concat(meat_antecedent_rules)
flat_meat_rules

final_rules=[]
for wine in all_wine:
    final_rules.append(
        flat_meat_rules[( flat_meat_rules['consequents'].astype(str).str.contains(wine) )]
        )

# flatten a list of dataframes into 1
flat_final_rules = pd.concat(final_rules)
len(flat_final_rules)
flat_final_rules.head(10)

# export the filtered rules out to better examine
flat_final_rules["antecedents"] = flat_final_rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
flat_final_rules["consequents"] = flat_final_rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")
flat_final_rules.to_csv('/Users/zhanyina/Documents/MSA/AA502 Analytics Methods and Applications I/Data Mining/Asgnmts/1/final_rules.csv')

# plot a heatmap of pairings, colored by confidence
pairings = pd.DataFrame(flat_final_rules, columns = ["antecedents", "consequents","confidence"])
pairings = pairings.pivot("antecedents", "consequents","confidence")
sns.heatmap(pairings, cmap="YlGnBu", linewidths=.5)
# annotate each cell with their confidence
sns.heatmap(pairings, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)

'''======= END market basket analysis on this data set ======='''





