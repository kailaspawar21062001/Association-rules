# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:07:06 2023

@author: kailas
"""

1]PROBLEM::--ASSOCIATION RULES FOR 'book.csv' dataset


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pip install "mlxtend"
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

#Dataset
data=pd.read_csv("D:/data science assignment/Assignments/9.Association rules/book.csv")
data

#EDA
data.head()
data.tail()
data.shape
data.describe()
data.isna().sum()


1. Association rules with 20% Support and 70% confidence
# With 20% Support
frequent_itemsets=apriori(data,min_support=0.2,use_colnames=True)
frequent_itemsets

# with 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules

rules.sort_values('lift',ascending=False)

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]

# visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


2. Association rules with 15% Support and 60% confidence
# With 15% Support
frequent_itemsets2=apriori(data,min_support=0.15,use_colnames=True)
frequent_itemsets2

# With 60% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

3. Association rules with 5% Support and 80% confidence
# With 5% Support
frequent_itemsets3=apriori(data,min_support=0.05,use_colnames=True)
frequent_itemsets3

# With 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3

rules3[rules3.lift>1]

# visualization of obtained rule
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()



#################################################################################
2]PROBLEM

::--ASSOCIATION RULES FOR 'my_movies.csv' dataset.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the Dataset
data = pd.read_csv("D:/data science assignment/Assignments/9.Association rules/my_movies.csv")

#EDA
data.shape
data.head()
data.info()
data1 = data.iloc[:,5:]
data1
data.describe()
data.isna().sum()


# Apriori Algorithm
1. Association rules with 10% Support and 70% confidence

from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

# with 10% support
frequent_itemsets = apriori(data1,min_support = 0.1,use_colnames=True)
frequent_itemsets

# 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules

# Lift Ratio>1 is a good influential rule is selecting the associated
rules[rules.lift>1]

# Visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

2. Association rules with 5% Support and 90% confidence
# with 5% support
frequent_itemsets2=apriori(data1,min_support=0.05,use_colnames=True)
frequent_itemsets2

# 90% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.9)
rules2

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules2[rules2.lift>1]

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


3. Association rules with 10% Support and 70% confidence
# with 10% support
frequent_itemsets2=apriori(data1,min_support=0.10,use_colnames=True)
frequent_itemsets2

# 70% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.7)
rules2

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules2[rules2.lift>1]

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()