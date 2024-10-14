#!/usr/bin/env python
# coding: utf-8

# <b> Line-level Vulnerability Detection</b>

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# Import libraries
import seaborn as sn
import pandas as pd
import json, os
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from collections import defaultdict
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

from transformers import set_seed
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForSequenceClassification #, BertModel, BertTokenizer

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

import logging


# Basic Configuration of logging and seed

# In[2]:


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# Define logger
logger = logging.getLogger(__name__)

# Specify a constant seeder for processes
seeders = [123456, 789012, 345678, 901234, 567890, 123, 456, 789, 135, 680]
seed = seeders[0]
logger.info(f"SEED: {seed}")
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
set_seed(seed)

checkpoint_dir = './checkpoints_seq2seq'
save_path = os.path.join(checkpoint_dir, 'best_weights.pt')


# Data Processing

# In[3]:


# Read dataset
root_path = os.getcwd()
dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))
dataset = dataset.dropna(subset=["processed_func"])


# In[4]:


# data split
val_ratio = 0.1
num_of_ratio = int(val_ratio * len(dataset))
data = dataset.iloc[0:-num_of_ratio, :]
test_data = dataset.iloc[-num_of_ratio:, :]
train_data = data.iloc[0:-num_of_ratio, :]
val_data = data.iloc[-num_of_ratio:, :]


# In[5]:


## train data
train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True) # shuffle training data

word_counts = train_data["processed_func"].apply(lambda x: len(x.split()))
max_length = word_counts.max()
logger.info(f"Maximum number of words: {max_length}")

# keep only vulnerable samples
train_data = train_data[train_data["target"] == 1]
train_data = train_data[~train_data['flaw_line_index'].isna()] # drop nan samples

# keep the useful for Seq2Seq columns
train_data = train_data[["processed_func", "flaw_line", "flaw_line_index"]]
train_data = train_data.reset_index(drop=True)

train_data = pd.DataFrame(({'Text': train_data['processed_func'], 'Lines':train_data['flaw_line'], 'Line_Index':train_data['flaw_line_index']}))

## validation data
# keep only vulnerable samples
val_data = val_data[val_data["target"] == 1]
val_data = val_data[~val_data['flaw_line_index'].isna()] # drop nan samples

# keep the useful for Seq2Seq columns
val_data = val_data[["processed_func", "flaw_line", "flaw_line_index"]]
val_data = val_data.reset_index(drop=True)

val_data = pd.DataFrame(({'Text': val_data['processed_func'], 'Lines':val_data['flaw_line'], 'Line_Index':val_data['flaw_line_index']}))

## test data
# keep only vulnerable samples
test_data = test_data[test_data["target"] == 1]
test_data = test_data[~test_data['flaw_line_index'].isna()] # drop nan samples

# keep the useful for Seq2Seq columns
test_data = test_data[["processed_func", "flaw_line", "flaw_line_index"]]
test_data = test_data.reset_index(drop=True)

test_data = pd.DataFrame(({'Text': test_data['processed_func'], 'Lines':test_data['flaw_line'], 'Line_Index':test_data['flaw_line_index']}))

# logs
logger.info(f"Train data length: {len(train_data)}")
logger.info(f"Validation data length: {len(val_data)}")
logger.info(f"Test data length: {len(test_data)}")

train_data.head()

# release some memory
del dataset






