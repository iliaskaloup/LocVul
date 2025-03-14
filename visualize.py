#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np


# In[2]:


width = 0.27  # Width of the bars


# In[3]:


# Chart for A@10
with plt.style.context('bmh'):
    x = np.array([0, 1])
    values = [82.8, 71.4]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('A@10 (%)', fontsize=20)
    plt.title('A@10', fontsize=18)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[4]:


# Chart for P@10
with plt.style.context('bmh'):
    x = np.array([0, 1])
    values = [26.9, 19.0]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('P@10 (%)', fontsize=20)
    plt.title('P@10', fontsize=18)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f'{value}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[5]:


# Chart for R@10
with plt.style.context('bmh'):
    x = np.array([0, 1])
    values = [79.0, 57.7]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('R@10 (%)', fontsize=20)
    plt.title('R@10', fontsize=18)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[7]:


# Chart for MRR@10
with plt.style.context('bmh'):
    x = np.array([0, 1])
    values = [79.4, 43.6]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('MRR@10 (%)', fontsize=20)
    plt.title('MRR@10', fontsize=18)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[8]:


# Chart for MAP@10
with plt.style.context('bmh'):
    x = np.array([0, 1])
    values = [79.2, 41.1]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('MAP@10 (%)', fontsize=20)
    plt.title('MAP@10', fontsize=18)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[9]:


with plt.style.context('bmh'):
    x = np.array([0, 1])
    values = [0, 2]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('Median IFA', fontsize=20)
    plt.title('IFA', fontsize=18)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(range(0, max(values) + 1), fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{value}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[10]:


# Box plot for IFA
import pandas as pd
ifa_self_attention = pd.read_csv('ifa_self_attention.csv')
ifa_locvul = pd.read_csv('ifa_locvul.csv')

with plt.style.context('bmh'):
    data = [
        ifa_locvul["IFA"].tolist(),  # IFA values for LocVul
        ifa_self_attention["IFA"].tolist()   # IFA values for Self-Attention
    ]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='darkblue', color='black', linewidth=2),
                medianprops=dict(color='red', linewidth=2), whiskerprops=dict(color='black', linewidth=2), capprops=dict(color='black', linewidth=2), widths=0.3)
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('IFA', fontsize=20)
    plt.title('Distribution of IFA', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# In[11]:


# bar chart with error bars for ifa - st. deviation

# Load data from CSV files
ifa_self_attention = pd.read_csv('ifa_self_attention.csv')
ifa_locvul = pd.read_csv('ifa_locvul.csv')

# Calculate mean and standard deviation for each method
mean_locvul = ifa_locvul['IFA'].mean()
std_locvul = ifa_locvul['IFA'].std(ddof=1)  # Sample standard deviation

mean_self_attention = ifa_self_attention['IFA'].mean()
std_self_attention = ifa_self_attention['IFA'].std(ddof=1)

# Prepare data for plotting
methods = ['LocVul', 'Self-Attention']
means = [mean_locvul, mean_self_attention]
std_devs = [std_locvul, std_self_attention]
x = np.arange(len(methods))  # X positions for bars
width = 0.4  # Adjusted width of bars for better spacing

# Create the bar chart with error bars
with plt.style.context('bmh'):
    plt.figure(figsize=(6, 6))
    bars = plt.bar(x, means, width, yerr=std_devs, capsize=8, 
                   color='darkblue', edgecolor='black', alpha=0.8, 
                   error_kw={'elinewidth':2, 'ecolor':'black'})
    
    # Add labels and title
    plt.xlabel('Methods', fontsize=14)
    plt.ylabel('Mean IFA', fontsize=14)
    plt.title('IFA Comparison', fontsize=16)
    
    # Set the x-ticks to the method labels
    plt.xticks(x, methods, fontsize=12)
    
    # Set y-ticks with appropriate range and reduce font size
    y_max = max(means) + max(std_devs) + 5
    plt.yticks(np.arange(0, y_max, step=5), fontsize=12)
    
    # Add grid for better readability
    plt.grid(visible=True, linestyle='--', alpha=0.6, axis='y')
    
    # Add value labels on top of each bar
    for bar, mean, std in zip(bars, means, std_devs):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + std + 1, 
                 f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Adjust y-axis limits to accommodate error bars
    plt.ylim(0, y_max)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the plot
    plt.show()


# In[14]:


# bar chart with error bars for ifa - iqr

# Load data
ifa_self_attention = pd.read_csv('ifa_self_attention.csv')
ifa_locvul = pd.read_csv('ifa_locvul.csv')

# Function to calculate median and IQR
def calculate_median_iqr(data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return median, iqr

# Calculate median and IQR for both methods
median_locvul, iqr_locvul = calculate_median_iqr(ifa_locvul['IFA'])
median_self_attention, iqr_self_attention = calculate_median_iqr(ifa_self_attention['IFA'])
print(iqr_self_attention/2 + median_self_attention)
print(iqr_locvul/2 + median_locvul)

# Prepare data for plotting
methods = ['LocVul', 'Self-Attention']
medians = [median_locvul, median_self_attention]
iqr_errors = [iqr_locvul / 2, iqr_self_attention / 2]  # Half IQR for error bars
x = np.arange(len(methods))

# Create the bar chart
with plt.style.context('bmh'):
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, medians, width, yerr=iqr_errors, capsize=8, 
                   color='darkblue', edgecolor='black', alpha=1.0,  # Fully opaque
                   error_kw={'elinewidth': 2, 'ecolor': 'black'})
    
    # Add labels and title
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('Median IFA', fontsize=20)
    plt.title('IFA', fontsize=18)
    
    # Set the x-ticks to the method labels
    plt.xticks(x, methods, fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add value labels on top of each bar
    for bar, median, iqr in zip(bars, medians, iqr_errors):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + iqr + 0.1, 
                 f'{int(round(median))}', ha='center', va='bottom', fontsize=12)
    
    # Adjust y-axis limits for better spacing
    y_max = max(medians) + max(iqr_errors) + 5
    plt.ylim(0, y_max)
    
    # Add grid for better readability
    plt.grid(visible=True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    plt.show()


# In[13]:


np.percentile(ifa_self_attention, 25)


# In[15]:


# Chart for Effort@20%Recall
with plt.style.context('bmh'):
    x = np.array([0, 1])
    values = [0.57, 0.61]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('Effort@20%Recall (%)', fontsize=20)
    plt.title('Effort@20%Recall', fontsize=18)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    for bar, value in zip(bars, values):
        #plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value}', ha='center', fontsize=12)
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{value}', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[16]:


# Chart for Recall@1%LOC
with plt.style.context('bmh'):
    x = np.array([0, 1])
    values = [29.8, 28.6]
    labels = ['LocVul', 'Self-Attention']
    plt.figure(figsize=(4.5, 5))
    bars = plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.bar(x, values, width, color='darkblue', edgecolor='black')
    plt.xlabel('Methods', fontsize=20)
    plt.ylabel('Recall@1%LOC (%)', fontsize=20)
    plt.title('Recall@1%LOC', fontsize=18)
    plt.xticks(x, labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    for bar, value in zip(bars, values):
        #plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value}', ha='center', fontsize=12)
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{value}', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.show()


# In[ ]:




