#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import wilcoxon
import statistics


# In[2]:


attention = os.path.join("results", "seeds", "attention")


# In[3]:


locvul = os.path.join("results", "seeds", "locvul")


# In[4]:


locvul_hal = os.path.join("results", "seeds", "locvul_hal")


# In[5]:


all_attention = pd.DataFrame()

# Walk through the directory and its subdirectories
frames = {}
for root, dirs, files in os.walk(attention):
    for file in files:
        # Check if the file is a CSV file
        if file.endswith(".csv"):
            # Construct the full path of the CSV file
            file_path = os.path.join(root, file)
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            frames[file] = df

all_attention = pd.concat(frames, names=['seed'])
all_attention.index = all_attention.index.get_level_values('seed')

print(all_attention)


# In[6]:


print(all_attention.describe())


# In[7]:


all_locvul = pd.DataFrame()

# Walk through the directory and its subdirectories
frames = {}
for root, dirs, files in os.walk(locvul):
    for file in files:
        # Check if the file is a CSV file
        if file.endswith(".csv"):
            # Construct the full path of the CSV file
            file_path = os.path.join(root, file)
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            frames[file] = df
            
all_locvul = pd.concat(frames, names=['seed'])
all_locvul.index = all_locvul.index.get_level_values('seed')

print(all_locvul)


# In[8]:


print(all_locvul.describe())


# In[9]:


all_locvul_hal = pd.DataFrame()

# Walk through the directory and its subdirectories
frames = {}
for root, dirs, files in os.walk(locvul_hal):
    for file in files:
        # Check if the file is a CSV file
        if file.endswith(".csv"):
            # Construct the full path of the CSV file
            file_path = os.path.join(root, file)
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            frames[file] = df
            
all_locvul_hal = pd.concat(frames, names=['seed'])
all_locvul_hal.index = all_locvul_hal.index.get_level_values('seed')

print(all_locvul_hal)


# In[10]:


print(all_locvul_hal.describe())


# In[11]:


var_of_interest = "accuracy" # accuracy, precision, recall, MRR, MAP, MAR, Median_IFA, EffortRecall, RecallLoc
#print(all_locvul[var_of_interest])


# Conduct the Wilcoxon-Signed Rank Test: pvalue < 0.05 --> statistically significant differentiation of the results

# In[12]:


## RQs
# Define the data arrays
approach1 = all_locvul[var_of_interest].values.tolist()
approach2 = all_attention[var_of_interest].values.tolist()

# Null hypothesis: approach1 is equal to approach 2 i.e., approach1 is not greater (or less for ifa and effortRecall) from approach2

# Alternative hypothesis: approach1 is greater (or less in for ifa and effortRecall) from approach2

# Conduct a one-tailed Wilcoxon signed-rank test
statistic, pvalue = wilcoxon(approach1, approach2, alternative='greater') # greater for all metrics except IFA and EffortRecall

print("Wilcoxon statistic coefficient:", statistic)
print("p-value:", pvalue)

if pvalue < 0.05:
    print("statistically significant differentiation")
else:
    print("no statistical significance")


# In[13]:


## Hallucinations
# Define the data arrays
approach1 = all_locvul[var_of_interest].values.tolist()
approach2 = all_locvul_hal[var_of_interest].values.tolist()

# Null hypothesis: approach1 is equal to approach 2 i.e., approach1 is not greater (or less for ifa and effortRecall) from approach2

# Alternative hypothesis: approach1 is greater (or less in for ifa and effortRecall) from approach2

# Conduct a one-tailed Wilcoxon signed-rank test
statistic, pvalue = wilcoxon(approach1, approach2, alternative='greater') # greater for all metrics except IFA and EffortRecall

print("Wilcoxon statistic coefficient:", statistic)
print("p-value:", pvalue)

if pvalue < 0.05:
    print("statistically significant differentiation")
else:
    print("no statistical significance")

