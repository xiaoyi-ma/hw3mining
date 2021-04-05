#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules


# Given the above transaction data t1 to t7, assume: minsup = 30%, minconf = 80%, find
# all the frequent itemset, calculate support

# In[2]:


dataset = [['Beef', 'Chicken', 'Milk'],
           ['Beef', 'Cheese'],
           ['Cheese', 'Boots'],
           ['Beef', 'Chicken', 'Cheese'],
           ['Beef', 'Chicken', 'Clothes', 'Cheese', 'Milk'],
           ['Chicken','Clothes','Milk'],
           ['Chicken','Milk','Clothes']]


# In[3]:


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
frequent_itemsets


# generate association rules from the itemset,
# calculate support and confidence

# In[5]:


association_rules(frequent_itemsets, metric="confidence", min_threshold=0.79999)

