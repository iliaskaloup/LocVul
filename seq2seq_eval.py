#!/usr/bin/env python
# coding: utf-8

# <b>Evaluation Scheme for line-level Vulnerability Detection using Seq2Seq models</b>

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
from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, \
roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

import logging
import statistics
import math
from sklearn.metrics.pairwise import cosine_similarity


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


# Read data and model

# In[3]:


# Read dataset
root_path = os.getcwd()
dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))


# In[4]:


checkpoint_dir = './checkpoints'
save_path = os.path.join(checkpoint_dir, 'best_weights.pt')

max_len_lines = 512
checkpoint_dir_seq2seq = './checkpoints_seq2seq_512'  # './checkpoints_seq2seq' for max_len_lines = 128 or './checkpoints_seq2seq_512' for max_len_lines = 512
save_path_seq2seq = os.path.join(checkpoint_dir_seq2seq, 'best_weights.pt')


# Get tokenizer

# In[5]:


model_variation = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True)


# In[6]:


model_variation_seq2seq = "Salesforce/codet5-base"
tokenizer_seq2seq = AutoTokenizer.from_pretrained(model_variation_seq2seq, do_lower_case=True)


# Split data sets and explore data

# In[7]:


# View the largest projects

logger.info(f"List of projects in BigVul: {dataset['project'].unique()}")
logger.info(f"Number of different projects in BigVul: {len(dataset['project'].unique())}")

project_counts = dataset['project'].value_counts().nlargest(10)
logger.info(f"Top-10 largest projects in BigVul and their size: {project_counts}")

# Choose the selected project to include in the test set.
# default = "all"
selected_project = "openssl" # all # Chrome # linux # Android # qemu # php # ImageMagick # savannah # FFmpeg # ghostscript # openssl


# In[8]:


# data split
val_ratio = 0.1
num_of_ratio = int(val_ratio * len(dataset))
data = dataset.iloc[0:-num_of_ratio, :]
test_data = dataset.iloc[-num_of_ratio:, :]
train_data = data.iloc[0:-num_of_ratio, :]
val_data = data.iloc[-num_of_ratio:, :]

# if selected_project=="all" continue with the whole test_set, else, if one specific project is selected keep only its samples
if selected_project != "all":
    test_data = test_data[test_data['project'] == selected_project]

# Shuffle dataset
train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
logger.info(f"Train data head: {train_data.head()}")
logger.info(f"Length of training data: {len(train_data)}")

train_data = train_data[["processed_func", "target", "flaw_line", "flaw_line_index"]]

# Explore data
train_data = train_data.dropna(subset=["processed_func"])

word_counts = train_data["processed_func"].apply(lambda x: len(x.split()))
max_length = word_counts.max()
logger.info(f"Maximum number of words: {max_length}")

vc = train_data["target"].value_counts()

logger.info(f"Value counts of training data: {vc}")

logger.info(f"Percentages of classes: {(vc[1] / vc[0])*100, '%'}")

n_categories = len(vc)
logger.info(f"Number of categories: {n_categories}")

train_data = pd.DataFrame(({'Text': train_data['processed_func'], 'Labels': train_data['target'], 'Lines':train_data['flaw_line'], 'Line_Index':train_data['flaw_line_index']}))
#train_data = train_data[0:100]
train_data.head()

val_data = pd.DataFrame(({'Text': val_data['processed_func'], 'Labels': val_data['target'], 'Lines':val_data['flaw_line'], 'Line_Index':val_data['flaw_line_index']}))
val_data.head()

test_data = pd.DataFrame(({'Text': test_data['processed_func'], 'Labels': test_data['target'], 'Lines':test_data['flaw_line'], 'Line_Index':test_data['flaw_line_index']}))

logger.info(f"Train data length: {len(train_data)}")
logger.info(f"Validation data length: {len(val_data)}")
logger.info(f"Test data length: {len(test_data)}")

del dataset


# Pre-processing

# In[9]:


# Pre-processing step: Under-sampling

sampling = False
if n_categories == 2 and sampling == True:
    # Apply under-sampling with the specified strategy
    class_counts = pd.Series(train_data["Labels"]).value_counts()
    print("Class distribution ", class_counts)

    majority_class = class_counts.idxmax()
    print("Majority class ", majority_class)

    minority_class = class_counts.idxmin()
    print("Minority class ", minority_class)

    target_count = 4 * class_counts[class_counts.idxmin()] # int(class_counts[class_counts.idxmax()] / 2) # 2 * class_counts[class_counts.idxmin()] # class_counts[class_counts.idxmin()] # int(class_counts.iloc[0] / 2)
    print("Targeted number of majority class", target_count)

    # under
    sampling_strategy = {majority_class: target_count}
    rus = RandomUnderSampler(random_state=seed, sampling_strategy=sampling_strategy)

    x_train_resampled, y_train_resampled = rus.fit_resample(np.array(train_data["Text"]).reshape(-1, 1), train_data["Labels"])
    print("Class distribution after augmentation", pd.Series(y_train_resampled).value_counts())


    # Shuffle the resampled data while preserving the correspondence between features and labels
    x_train_resampled, y_train_resampled = shuffle(x_train_resampled, y_train_resampled, random_state=seed)

    # rename
    X_train = x_train_resampled
    Y_train = y_train_resampled

    X_train = pd.Series(X_train.reshape(-1))

else:
    X_train = train_data["Text"]
    Y_train = train_data["Labels"]


# Get model and apply tokenizer

# In[10]:


# Pre-trained model

model = AutoModelForSequenceClassification.from_pretrained(model_variation, num_labels=n_categories)
# Resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)


# # Compute maximum length

# X = tokenizer(
#         text=X_train.tolist(),
#         add_special_tokens=True,
#         max_length=512,
#         truncation=True,
#         padding=True,
#         return_tensors='pt',
#         return_token_type_ids=False,
#         return_attention_mask=True,
#         verbose=True
#     )

# max_len = getMaxLen(X)
max_len = 512

# Tokenization

X_train = tokenizer(
    text=X_train.tolist(),
    add_special_tokens=True,
    max_length=max_len,
    truncation=True,
    padding=True,
    return_tensors='pt',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)


X_val = tokenizer(
    text=val_data['Text'].tolist(),
    add_special_tokens=True,
    max_length=max_len,
    truncation=True,
    padding=True,
    return_tensors='pt',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)


X_test = tokenizer(
    text=test_data['Text'].tolist(),
    add_special_tokens=True,
    max_length=max_len,
    truncation=True,
    padding=True,
    return_tensors='pt',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)


# Model preparation

# In[11]:


# Hyper-parameters

n_epochs = 10
lr = 2e-5 #5e-05
batch_size = 8 #16
patience = 5

optimizer = AdamW(model.parameters(),
                  lr = lr, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )


# In[12]:


# Build Model

Y_train = torch.LongTensor(Y_train.tolist())
Y_val = torch.LongTensor(val_data["Labels"].tolist())
Y_test = torch.LongTensor(test_data["Labels"].tolist())
Y_train.size(), Y_val.size(), Y_test.size()


train_dataset = TensorDataset(X_train["input_ids"], X_train["attention_mask"], Y_train)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

val_dataset = TensorDataset(X_val["input_ids"], X_val["attention_mask"], Y_val)
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

test_dataset = TensorDataset(X_test["input_ids"], X_test["attention_mask"], Y_test)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)


max_steps = len(train_dataloader)*n_epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=max_steps // 5,
            num_training_steps=max_steps)

loss_fun = nn.CrossEntropyLoss()

# total_steps = len(train_dataloader) * n_epochs

# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, # Default value in run_glue.py
#                                             num_training_steps = total_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device {device}")

print(model.to(device))
print("No. of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[13]:


# Function to replace "/~/" with "\n" in the 'Lines' column
def replace_delimiter_with_newline(data):
    # Replace "/~/" with "\n" in the 'Lines' column
    data['Lines'] = data['Lines'].str.replace('/~/', '\n')
    return data

test_data = replace_delimiter_with_newline(test_data)


# Execution loop

# In[14]:


# Load model
checkpoint = torch.load(save_path, map_location=device)
# If model is wrapped in DataParallel, load state_dict directly into the underlying model
if torch.cuda.device_count() > 1:
    model.module.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint['model'])
model.to(device)

# Eliminate Test samples that are vulnerable (target=1) but they have missing line-level labels (flaw lines is nan)
REMOVE_MISSING_LINE_LABELS = True # True # False

if REMOVE_MISSING_LINE_LABELS:

    test_data = test_data.reset_index(drop=True)
    test_data = test_data[~((test_data['Labels'] == 1) & (test_data['Line_Index'].isna()))]
    test_data = test_data.reset_index(drop=True)
    Y_test = torch.LongTensor(test_data["Labels"].tolist())
    
    
    X_test = tokenizer(
        text=test_data['Text'].tolist(),
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors='pt',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True
    )
    
    test_dataset = TensorDataset(X_test["input_ids"], X_test["attention_mask"], Y_test)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
    
    # Make predictions
    logger.info("Starting testing...")
    test_start_time = time.time()
    model.eval()
    test_pred = []
    test_probas_pred = []
    actual_labels = []
    test_loss = 0
    with torch.no_grad():
        for step_num, batch_data in enumerate(tqdm(test_dataloader, desc='Testing')):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
    
            output = model(input_ids = input_ids, attention_mask=att_mask) #, labels= labels
    
            loss = loss_fun(output.logits, labels) #loss = output.loss #output[0]
            test_loss += loss.item()
    
            logits_array = output.logits.cpu().detach().numpy()
            #probs_array = softmax(logits_array, axis=1)
            probs_array = torch.softmax(torch.tensor(logits_array), dim=-1).numpy()
            
            preds = np.argmax(probs_array , axis=-1)
            test_pred+=list(preds)
            actual_labels+=labels.cpu().numpy().tolist()
    
            probas = np.max(probs_array , axis=1)
            test_probas_pred+=list(probas)
    
    # compute evaluation metrics
    new_class_report = classification_report(actual_labels, test_pred)
    logger.info(f"Classification Report:\n{new_class_report}")


# In[15]:


# Identify negative predictions ie TN and FN
negative_indices = [i for i, pred in enumerate(test_pred) if pred == 0]  # Indices of Negative predictions (TNs + FNs)

# Collect lines of negative predictions
negative_samples = [test_data['Text'].tolist()[i] for i in negative_indices]  # Extract Negative samples from test data

# Flatten
all_neg_lines = []
for neg_func in negative_samples:
    neg_lines = neg_func.split('\n') #function_to_lines
    for neg_line in neg_lines:
        all_neg_lines.append(neg_line)


# In[16]:


ONLY_TP_Accuracy = True
ONLY_TP_CostEffect = False

ONLY_TP = ONLY_TP_CostEffect


# In[17]:


# Identify True Positives (where the predicted label and actual label are both 1)
true_positive_indices = [i for i, (pred, label) in enumerate(zip(test_pred, Y_test.tolist())) if pred == 1 and label == 1]
if ONLY_TP:
    positive_indices = true_positive_indices
    logger.info(f"Selected {len(true_positive_indices)} True Positives for explanations.")
else:
    # Identify True Positives and False Positives
    trueNfalse_positive_indices = [i for i, pred in enumerate(test_pred) if pred == 1]  # Indices of Positive predictions (TPs + FPs)
    logger.info(f"Generating explanations for {len(trueNfalse_positive_indices)} Positive predictions (TPs and FPs)...")
    positive_indices = trueNfalse_positive_indices

actual_positive_indices = [i for i, label in enumerate(Y_test.tolist()) if label == 1]  # Indices of Actual Positive predictions (TPs + FNs)


# In[18]:


positive_samples = [test_data['Text'].tolist()[i] for i in positive_indices]  # Extract Positive samples from test data

positive_lines = [test_data['Lines'].tolist()[i] for i in positive_indices]
positive_lines = ["" if isinstance(x, float) and math.isnan(x) else x for x in positive_lines]

positive_probas = [test_probas_pred[i] for i in positive_indices]


# Apply Seq2Seq model

# In[19]:


# def tokenize_data_without_labels(tokenizer, positive_samples):
#     input_encodings = tokenizer(
#         positive_samples,
#         max_length=512,
#         truncation=True,
#         padding='max_length',
#         return_tensors='pt',
#         add_special_tokens=True
#     )
    
#     return input_encodings

# # Tokenize the test data without labels
# test_encodings = tokenize_data_without_labels(tokenizer_seq2seq, positive_samples)

# # Create a TensorDataset only with input_ids and attention_mask (no labels)
# test_dataset_seq2seq = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
# test_loader_seq2seq = DataLoader(test_dataset_seq2seq, sampler=SequentialSampler(test_dataset_seq2seq), batch_size=batch_size)


# In[20]:


def tokenize_data(tokenizer, positive_samples, positive_lines, max_len_lines):
    input_encodings = tokenizer(
        positive_samples,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        add_special_tokens=True
    )
    
    target_encodings = tokenizer(
        positive_lines,
        max_length=max_len_lines,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        add_special_tokens=True
    )

    input_encodings['labels'] = target_encodings['input_ids']
    
    return input_encodings

test_encodings = tokenize_data(tokenizer_seq2seq, positive_samples, positive_lines, max_len_lines)
test_dataset_seq2seq = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_encodings['labels'])
test_loader_seq2seq = DataLoader(test_dataset_seq2seq, sampler=SequentialSampler(test_dataset_seq2seq), batch_size=batch_size)


# In[21]:


# Load the CodeT5 model
model_seq2seq = AutoModelForSeq2SeqLM.from_pretrained(model_variation_seq2seq)

#load model
checkpoint = torch.load(save_path_seq2seq, map_location=device)
# If model is wrapped in DataParallel, load state_dict directly into the underlying model
if torch.cuda.device_count() > 1:
    model_seq2seq.module.load_state_dict(checkpoint['model'])
else:
    model_seq2seq.load_state_dict(checkpoint['model'])
print(model_seq2seq.to(device))


# In[22]:


# Make predictions on the testing set
logger.info("Starting testing...")
test_start_time = time.time()

model.eval()
test_preds = []
actual_labels = []
test_loss = 0

with torch.no_grad():
    for step_num, batch_data in enumerate(tqdm(test_loader_seq2seq, desc='Testing')):
        input_ids, attention_mask, labels = [data.to(device) for data in batch_data]

        # Generate predictions
        if torch.cuda.device_count() > 1:
            outputs = model_seq2seq.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_len_lines)
        else:
            outputs = model_seq2seq.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_len_lines)
        
        # Decode predicted sequences and actual labels
        decoded_preds = tokenizer_seq2seq.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer_seq2seq.batch_decode(labels, skip_special_tokens=True)

        test_preds.extend(decoded_preds)
        actual_labels.extend(decoded_labels)

test_end_time = time.time()
testing_time = test_end_time - test_start_time

# Display the total testing time and average time per sample
print("Testing completed after", testing_time)
print("Perception time per sample:", int(testing_time / len(test_preds)))


# In[23]:


# compute the average number of lines predicted as vulnerable by the seq2seq model
pred_lens = []
for i in range(0, len(test_preds)):
    pred_lens.append(len(test_preds[i].split('\n')))
mean_pred_len = statistics.mean(pred_lens)
med_pred_len = statistics.median(pred_lens)
logger.info(f"Mean predicted length: {mean_pred_len}")
logger.info(f"Median predicted length: {med_pred_len}")


# In[24]:


# compute the average number of lines that are actual vulnerable lines
actual_vuln_lens = []
for i in range(0, len(positive_lines)):
    actual_vuln_lens.append(len(positive_lines[i].split('\n')))
mean_actual_vuln_len = statistics.mean(actual_vuln_lens)
med_actual_vuln_len = statistics.median(actual_vuln_lens)
logger.info(f"Mean actual flaw length: {mean_actual_vuln_len}")
logger.info(f"Median actual flaw length: {med_actual_vuln_len}")


# In[25]:


def calc_accurary(test_preds, real_positive_lines):
    accuracy = 0
    for i in range(0, len(test_preds)):
        if test_preds[i] == real_positive_lines[i]:
            accuracy += 1
    accuracy = accuracy / len(test_preds)
    return accuracy


# In[26]:


# compute simple accuracy: In how many functions the seq2seq model identified the vulnerable lines 100%
accuracy = calc_accurary(test_preds, positive_lines)
logger.info(f"Accuracy: {accuracy*100, '%'}")


# In[27]:


# compute simple accuracy with truncated output: In how many functions the seq2seq model identified the vulnerable lines 100%, 
#considering the actual vulnerable lines truncated in max_len
accuracy_trunc = calc_accurary(test_preds, actual_labels)
logger.info(f"Accuracy on truncated labels: {accuracy_trunc*100, '%'}")


# In[28]:


# compute accuracy metrics using the most similar lines of the predicted to handle hallucinations

def get_line_embeddings(lines, tokenizer, model):
    """
    Get the embeddings for a list of lines using a CodeT5 model.
    
    Args:
    lines (list of str): The lines of code to embed.
    tokenizer: The tokenizer for the CodeT5 model.
    model: The CodeT5 model.
    
    Returns:
    embeddings (torch.Tensor): A tensor containing the embeddings for each line.
    """
    # Tokenize the input lines
    inputs = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get the model output
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    
    # Extract the last hidden state
    hidden_states = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
    
    # To get a single embedding per line, we can mean-pool the hidden states across the sequence dimension
    # or use just the first token's representation, depending on your task.
    # Here, we'll use mean-pooling:
    embeddings = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_dim)
    
    return embeddings.cpu().numpy()  # Return the embeddings as a NumPy array
    

def get_most_similar_line(predicted_line, original_lines, tokenizer, model):
    """
    Find the most similar line from original lines based on cosine similarity.
    """
    
    predicted_embedding = get_line_embeddings([predicted_line], tokenizer, model)[0]
    original_embeddings = get_line_embeddings(original_lines, tokenizer, model)

    cosine_similarities = cosine_similarity([predicted_embedding], original_embeddings).flatten()
    
    # lines = [predicted_line] + original_lines  # Combine predicted with original lines
    
    # # Compute TF-IDF matrix
    # vectorizer = TfidfVectorizer().fit_transform(lines)
    # vectors = vectorizer.toarray()
    
    # # Calculate cosine similarity between the first line (predicted) and the rest
    # cosine_similarities = cosine_similarity([vectors[0]], vectors[1:])
    
    most_similar_idx = np.argmax(cosine_similarities)  # Find the index of the most similar line
    
    return original_lines[most_similar_idx]


# Choose whether to replace predicted lines with the most similar lines in the function to handle hallucinations

# In[29]:


SIMILARITY_REPLACEMENT = False

if SIMILARITY_REPLACEMENT:

    test_preds_similar = []
    for i, pred in enumerate(test_preds):
        predicted_lines  = pred.split('\n')
        original_lines  = positive_samples[i].split('\n')
        similar_str = ''
        for j, predicted_line in enumerate(predicted_lines):
            if predicted_line not in original_lines and j < len(predicted_lines)-1: # to avoid to spoil a correct line AND to just to avoid a difference with the actual_labels in the evaluation
                similar_line = get_most_similar_line(predicted_line, original_lines, tokenizer_seq2seq, model_seq2seq)
            else:
                similar_line = predicted_line
            if j == 0:
                similar_str += similar_line
            else:
                similar_str += '\n' + similar_line
                
        test_preds_similar.append(similar_str) 
    
    accuracy_similar = calc_accurary(test_preds_similar, positive_lines)
    logger.info(f"Accuracy: {accuracy_similar*100, '%'}")
    
    accuracy_similar_trunc = calc_accurary(test_preds_similar, actual_labels)
    logger.info(f"Accuracy on truncated labels: {accuracy_similar_trunc*100, '%'}")
    
    test_preds = test_preds_similar


# In[32]:


num=100


# In[33]:


print(positive_samples[num])


# In[34]:


print(positive_lines[num])


# In[35]:


print(actual_labels[num])


# In[36]:


print(test_preds[num])


# In[37]:


print(test_preds[num] == actual_labels[num])


# In[38]:


print(test_preds[num] == positive_lines[num])


# Rank the lines based on the predictions of the seq2seq model and their position in the original functions

# In[39]:


all_ranked_lines = []
all_predicted_lines_number = []

# the ranked list per function will contain first the predicted lines by the seq2seq model and then the rest lines of the original function 
# with order based on their original position
all_predicted_lines = []
for pred in test_preds:
    predicted_lines = pred.split('\n')
    all_predicted_lines.append(predicted_lines)
    all_ranked_lines.append(predicted_lines.copy())
    all_predicted_lines_number.append(len(predicted_lines))

for i, pos_sample in enumerate(positive_samples):
    original_lines = pos_sample.split('\n')
    for orig_line in original_lines:
        if orig_line not in all_predicted_lines[i]:
            all_ranked_lines[i].append(orig_line)

# restructure labels
str_labels = positive_lines # actual_labels # positive_lines
all_flaw_lines = []
for label in str_labels:
    label_lines = label.split('\n')
    all_flaw_lines.append(label_lines)


# Line-level Evaluation

# In[40]:


# Accuracy metrics

# Function to compute Top-X Accuracy for each function
def compute_top_x_accuracy(ranked_lines, flaw_lines, top_x):
    """
    Compute Top-X Accuracy: Measures whether at least one actual vulnerable line appears in the top-X ranking.
    
    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices (integers).
    :param top_x: The number of top lines to consider (default is 10).
    :return: 1 if at least one vulnerable line is in the top-X, else 0.
    """
    top_x_lines = ranked_lines[:top_x]  # Get the top-X ranked lines

    return 1 if any(line in flaw_lines for line in top_x_lines) else 0


def compute_reciprocal_rank(ranked_lines, flaw_lines, top_x):
    """
    Compute Reciprocal Rank for a single function.

    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices (integers).
    :return: Reciprocal rank for this function, or 0 if no vulnerable line is found in the ranking.
    """

    top_x_lines = ranked_lines[:top_x]  # Get the top-X ranked lines
    for i, line in enumerate(top_x_lines):
        if line in flaw_lines:
            return 1 / (i + 1)  # Reciprocal of the rank of the first relevant item
    return 0  # If no relevant item is found
    

# Function to compute Initial False Alarm (IFA)
def compute_ifa(ranked_lines, flaw_lines):
    """
    Compute Initial False Alarm (IFA): Counts how many false alarms (non-vulnerable lines) occur before the first vulnerable line.
    
    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices.
    :return: Number of false alarms until the first vulnerable line is found.
    """
    ifa = 0
    for line in ranked_lines:
        if line not in flaw_lines:
            ifa += 1
        else:
            break  # Stop counting when the first vulnerable line is found
    return ifa


# In[41]:


# Function to compute Top-X Precision for each function
def compute_top_x_precision(ranked_lines, flaw_lines, top_x):
    """
    Compute Top-X Precision: Measures how many lines are indeed vulnerable in the top-X ranking.

    Relevant retrieved instances divided by all retrieved instances
    
    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices (integers).
    :param top_x: The number of top lines to consider (default is 10).
    :return: Number of the number of vulnerable lines included in the top-X ranking divided by X.
    """
    top_x_lines = ranked_lines[:top_x]  # Get the top-X ranked lines

    count = 0
    for line in top_x_lines:
        if line in flaw_lines:
            count += 1

    return count / top_x


# Function to compute Top-X Recall for each function
def compute_top_x_recall(ranked_lines, flaw_lines, top_x):
    """
    Compute Top-X Recall: Measures how many of the function's vulnerable lines can be found by searching in the top-X ranking.

    Relevant retrieved instances divided by all relevant instances
    
    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices (integers).
    :param top_x: The number of top lines to consider (default is 10).
    :return: Number of the number of vulnerable lines included in the top-X ranking divided by the total number of vulnerable lines in the function.
    """
    top_x_lines = ranked_lines[:top_x]  # Get the top-X ranked lines

    count = 0
    for line in top_x_lines:
        if line in flaw_lines:
            count += 1

    return count / len(flaw_lines)


# In[42]:


def compute_average_precision_at_k(ranked_lines, flaw_lines, k):
    """
    Compute Average Precision at K for a single function.

    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices (integers).
    :param k: The number of top lines to consider for AP@K.
    :return: Average Precision at K for this function.
    """
    relevant_found = 0
    precision_sum = 0
    top_k_lines = ranked_lines[:k]  # Consider only the top K lines

    for i, line in enumerate(top_k_lines):
        if line in flaw_lines:
            relevant_found += 1
            precision_sum += relevant_found / (i + 1)  # Precision at this rank

    return precision_sum / relevant_found if relevant_found>0 else 0  # Avoid division by zero
    #return precision_sum / min(k, len(flaw_lines)) if flaw_lines else 0  # Avoid division by zero


def compute_average_recall_at_k(ranked_lines, flaw_lines, k):
    """
    Compute Average Recall at K for a single function.

    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices (integers).
    :param k: The number of top lines to consider for AR@K.
    :return: Average Recall at K for this function.
    """
    relevant_found = 0
    precision_sum = 0
    top_k_lines = ranked_lines[:k]  # Consider only the top K lines

    for i, line in enumerate(top_k_lines):
        if line in flaw_lines:
            relevant_found += 1
            precision_sum += relevant_found / len(flaw_lines)  # Precision at this rank

    return precision_sum / relevant_found if relevant_found>0 else 0  # Avoid division by zero


# In[43]:


# Cost-Effectiveness metrics

# Helper functions
# Compute total LOC of the testing set
def compute_total_loc(all_total_locs):  
    return sum(all_total_locs)

def compute_total_flaw_lines(all_flaw_lines):
    total_flaw_loc = 0
    for flaw_lines in all_flaw_lines:
        total_flaw_loc+=len(flaw_lines)

    return total_flaw_loc

def find_effort_breakpoint(flaw_lines_num, x_percent):    
    return max(1, ((x_percent/100) * flaw_lines_num))

def find_recall_breakpoint(total_test_loc, x_percent):    
    return max(1, ((x_percent/100) * total_test_loc))

# Prepare data for Cost-Effectiveness calculation
# Sort the ranked_lines based on their function proba
def sort_all_ranked_lines(positive_probas, all_ranked_lines):
    combined = list(zip(positive_probas, all_ranked_lines))
    combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
    all_ranked_lines_sorted = [item[1] for item in combined_sorted]
    
    return all_ranked_lines_sorted

# # Sort the flaw_lines based on their function proba
# def sort_all_flaw_lines(positive_probas, all_flaw_lines):

#     combined = list(zip(positive_probas, all_flaw_lines))
#     combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
#     all_flaw_lines_sorted = [item[1] for item in combined_sorted]

#     return all_flaw_lines_sorted
    

# Function to compute Effort@X%Recall by sorting functions
def compute_effort_at_x_percent_recall_rankedFuncs(all_ranked_lines, positive_probas, all_flaw_lines, test_all_flaw_lines, all_total_locs, x_percent=20):

    # Prepare data for Cost-Effectiveness calculation
    all_ranked_lines_sorted = sort_all_ranked_lines(positive_probas, all_ranked_lines)
    #all_flaw_lines_sorted = sort_all_flaw_lines(positive_probas, all_flaw_lines)
    all_flaw_lines_sorted = sort_all_ranked_lines(positive_probas, all_flaw_lines)
    
    total_test_loc = compute_total_loc(all_total_locs)

    flaw_lines_num = compute_total_flaw_lines(test_all_flaw_lines)
    
    effort_breakpoint = find_effort_breakpoint(flaw_lines_num, x_percent)

    if flaw_lines_num == 0:
        return 1.0  # If no vulnerable lines, maximum effort (full LOC inspected)

    # Iterate over ranked lines to count how much effort (LOC) is spent to find X% of the vulnerable lines
    inspected_lines = 0
    found_vulnerable_lines = 0
    found = False
    for i, fun_lines in enumerate(all_ranked_lines_sorted):
        fun_flaws = all_flaw_lines_sorted[i]
        for line in fun_lines:
            inspected_lines += 1

            if line in fun_flaws:
                found_vulnerable_lines += 1

            # Stop when we find X% of vulnerable lines
            if found_vulnerable_lines >= effort_breakpoint:
                found = True
                break
        if found:
            break

    return inspected_lines / total_test_loc

# Assign labels for all sorted lines
def create_sorted_lines_with_labels(all_ranked_lines, all_flaw_lines, all_predicted_lines_number):

    all_lines_with_labels = []
    for func_idx, ranked_lines in enumerate(all_ranked_lines):
        flaw_lines = all_flaw_lines[func_idx]
        
        for line_idx, line_content in enumerate(ranked_lines):
            if line_idx < all_predicted_lines_number[func_idx]:
                line_score = 1 # lines predicted as vulnerable
            else:
                line_score = 0 # lines not predicted as vulnerable
                
            if line_content in flaw_lines:
                label = 1
            else:
                label = 0

            all_lines_with_labels.append((line_content, line_score, label))

    sorted_lines_with_labels = sorted(all_lines_with_labels, key=lambda x: x[1], reverse=True)
    
    return sorted_lines_with_labels

# Assign labels for all sorted lines and sorted functions
def create_sorted_lines_with_labels_and_probas(all_ranked_lines, positive_probas, all_flaw_lines, all_predicted_lines_number):

    all_ranked_lines_sorted = sort_all_ranked_lines(positive_probas, all_ranked_lines)
    all_flaw_lines_sorted = sort_all_ranked_lines(positive_probas, all_flaw_lines)
    all_predicted_lines_number_sorted = sort_all_ranked_lines(positive_probas, all_predicted_lines_number)
    
    all_lines_with_labels = []
    for func_idx, ranked_lines in enumerate(all_ranked_lines_sorted):
        flaw_lines = all_flaw_lines_sorted[func_idx]
        
        for line_idx, line_content in enumerate(ranked_lines):
            if line_idx < all_predicted_lines_number_sorted[func_idx]:
                line_score = 1 # lines predicted as vulnerable
            else:
                line_score = 0 # lines not predicted as vulnerable
                
            if line_content in flaw_lines:
                label = 1
            else:
                label = 0

            all_lines_with_labels.append((line_content, line_score, label))

    # # Separate lines with line_score=0 and line_score=1
    # vulnerable_lines = [line for line in all_lines_with_labels if line[1] == 1]
    # non_vulnerable_lines = [line for line in all_lines_with_labels if line[1] == 0]

    # # Shuffle only the non-vulnerable lines
    # random.shuffle(non_vulnerable_lines)

    # # Combine the shuffled non-vulnerable lines with the vulnerable lines
    # sorted_lines_with_labels = vulnerable_lines + non_vulnerable_lines
    
    sorted_lines_with_labels = sorted(all_lines_with_labels, key=lambda x: x[1], reverse=True)
    
    return sorted_lines_with_labels

# Function to compute Effort@X%Recall by sorting all lines
def compute_effort_at_x_percent_recall_rankedLines(all_ranked_lines, positive_probas, all_flaw_lines, test_all_flaw_lines, all_total_locs, all_predicted_lines_number, x_percent=20):
    
    # Prepare data for Cost-Effectiveness calculation
    #all_labels_lines_sorted = create_sorted_lines_with_labels(all_ranked_lines, all_flaw_lines, all_predicted_lines_number) # contains the label (vulnerable or not) of each line in the sorted lines
    all_labels_lines_sorted = create_sorted_lines_with_labels_and_probas(all_ranked_lines, positive_probas, all_flaw_lines, all_predicted_lines_number)
    
    total_test_loc = compute_total_loc(all_total_locs)

    flaw_lines_num = compute_total_flaw_lines(test_all_flaw_lines)

    if flaw_lines_num == 0:
        return 1.0  # If no vulnerable lines, maximum effort (full LOC inspected)

    effort_breakpoint = find_effort_breakpoint(flaw_lines_num, x_percent)

    # Iterate over ranked lines to count how much effort (LOC) is spent to find X% of the vulnerable lines
    inspected_lines = 0
    found_vulnerable_lines = 0
    for i in range(0, len(all_labels_lines_sorted)):
        _, _, line_label = all_labels_lines_sorted[i]
        inspected_lines += 1
        if line_label == 1:
            found_vulnerable_lines += 1

        # Stop when we find X% of vulnerable lines
        if found_vulnerable_lines >= effort_breakpoint:
            break

    return inspected_lines / total_test_loc
    

# Function to compute Recall@1%LOC by sorting functions
def compute_recall_at_x_percent_loc_rankedFuncs(all_ranked_lines, positive_probas, all_flaw_lines, test_all_flaw_lines, all_total_locs, x_percent=1):

    # Prepare data for Cost-Effectiveness calculation
    all_ranked_lines_sorted = sort_all_ranked_lines(positive_probas, all_ranked_lines)
    #all_flaw_lines_sorted = sort_all_flaw_lines(positive_probas, all_flaw_lines)
    all_flaw_lines_sorted = sort_all_ranked_lines(positive_probas, all_flaw_lines)
    
    total_test_loc = compute_total_loc(all_total_locs)

    flaw_lines_num = compute_total_flaw_lines(test_all_flaw_lines)
    
    recall_breakpoint = find_recall_breakpoint(total_test_loc, x_percent)

    # Count how many vulnerable lines are found within the top X% LOC
    inspected_lines = 0
    found_vulnerable_lines = 0
    found = False
    for i, fun_lines in enumerate(all_ranked_lines_sorted):
        fun_flaws = all_flaw_lines_sorted[i]
        for line in fun_lines:
            inspected_lines += 1

            if line in fun_flaws:
                found_vulnerable_lines += 1

            # Stop when we find X% of vulnerable lines
            if inspected_lines >= recall_breakpoint:
                found = True
                break
        if found:
            break

    return found_vulnerable_lines / flaw_lines_num

# Function to compute Recall@1%LOC by sorting all lines
def compute_recall_at_x_percent_loc_rankedLines(all_ranked_lines, positive_probas, all_neg_lines, all_flaw_lines, test_all_flaw_lines, all_total_locs, all_predicted_lines_number, x_percent=1):

    # Prepare data for Cost-Effectiveness calculation
    #all_labels_lines_sorted = create_sorted_lines_with_labels(all_ranked_lines, all_flaw_lines, all_predicted_lines_number) # contains the label (vulnerable or not) of each line in the sorted lines
    all_labels_lines_sorted = create_sorted_lines_with_labels_and_probas(all_ranked_lines, positive_probas, all_flaw_lines, all_predicted_lines_number)
    
    total_test_loc = compute_total_loc(all_total_locs)

    flaw_lines_num = compute_total_flaw_lines(test_all_flaw_lines)
    
    recall_breakpoint = find_recall_breakpoint(total_test_loc, x_percent)

    # Count how many vulnerable lines are found within the top X% LOC
    inspected_lines = 0
    found_vulnerable_lines = 0
    inspect_neg_lines = True
    for i in range(0, len(all_labels_lines_sorted)):
        inspected_lines += 1
        _, _, line_label = all_labels_lines_sorted[i]

        if line_label == 1:
            found_vulnerable_lines += 1

        if inspected_lines >= recall_breakpoint:
            inspect_neg_lines = False
            break

    if inspect_neg_lines:
        for neg_line in all_neg_lines:
            inspected_lines += 1
            if inspected_lines >= recall_breakpoint:
                break
            
    return found_vulnerable_lines / flaw_lines_num


# In[44]:


# Function to evaluate all metrics for each function
def evaluate_vulnerability_detection(all_ranked_lines, all_flaw_lines, top_x):
    """
    Evaluate the XAI methods using Top-X Accuracy, IFA, Effort@X%Recall, Recall@X%LOC for all functions.

    :param all_ranked_lines: List of ranked lines for all functions.
    :param all_flaw_lines: List of actual vulnerable line indices for all functions.
    :param top_x: Number of top-ranked lines to consider for Top-X Accuracy.
    :return: DataFrame with individual and average results for each function.
    """
    results = []
    for i, ranked_lines in enumerate(all_ranked_lines):
        
        flaw_lines = all_flaw_lines[i]
        
        # Compute each metric
        top_x_accuracy = compute_top_x_accuracy(ranked_lines, flaw_lines, top_x)

        top_x_precision = compute_top_x_precision(ranked_lines, flaw_lines, top_x)

        top_x_recall = compute_top_x_recall(ranked_lines, flaw_lines, top_x)

        rr = compute_reciprocal_rank(ranked_lines, flaw_lines, top_x)

        apk = compute_average_precision_at_k(ranked_lines, flaw_lines, top_x)

        ark = compute_average_recall_at_k(ranked_lines, flaw_lines, top_x)

        
        ifa = compute_ifa(ranked_lines, flaw_lines)

        result = {
            f'Top-{top_x} Accuracy': top_x_accuracy,
            f'Top-{top_x} Precision': top_x_precision,
            f'Top-{top_x} Recall': top_x_recall,
            f'Reciprocal Rank-{top_x}': rr,
            f'AP@{top_x}': apk,
            f'AR@{top_x}': ark,
            'IFA': ifa
        }
        
        results.append(result)


    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Compute average results
    average_results = results_df.mean().to_dict()
    average_results['Type'] = 'Average'

    # Compute median results
    median_results = results_df.median().to_dict()
    median_results['Type'] = 'Median'

    # Add individual results and average to the final DataFrame
    results_df['Type'] = 'Individual'
    
    average_results_df = pd.DataFrame([average_results])
    median_results_df = pd.DataFrame([median_results])

    # Combine individual and average results
    final_results_df = pd.concat([results_df, average_results_df, median_results_df], ignore_index=True)
    
    return final_results_df


# In[45]:


# Results based on per function accuracy

# Usage:
top_x = 10
final_results_df = evaluate_vulnerability_detection(all_ranked_lines, all_flaw_lines, top_x)

# Display Accuracy Results per Function
print(final_results_df)


# In[46]:


# Prepare data for line-level evaluation of cost-effectiveness
test_all_flaw_lines = [test_data['Line_Index'].tolist()[i] for i in actual_positive_indices] # Extract the flaw line indexes for each actual positive sample
test_all_total_locs = [len(test_data['Text'].tolist()[i].split('\n')) for i in range(len(test_data))] # Compute total LOC for each sample in the testing set


# In[47]:


# Results based on the total of lines

# configure sorting choice
sort_by_lines = True # False # True when sort lines by line score and False when sort functions by prediction proba (and then sort lines in each function)

# Usage
if sort_by_lines == False:
    effortXrecall = compute_effort_at_x_percent_recall_rankedFuncs(all_ranked_lines, positive_probas, all_flaw_lines, test_all_flaw_lines, test_all_total_locs, x_percent=20)
    recallXloc = compute_recall_at_x_percent_loc_rankedFuncs(all_ranked_lines, positive_probas, all_flaw_lines, test_all_flaw_lines, test_all_total_locs, x_percent=1)
else: #sort_by_lines == True
    effortXrecall = compute_effort_at_x_percent_recall_rankedLines(all_ranked_lines, positive_probas, all_flaw_lines, test_all_flaw_lines, test_all_total_locs, all_predicted_lines_number, x_percent=20)
    recallXloc = compute_recall_at_x_percent_loc_rankedLines(all_ranked_lines, positive_probas, all_neg_lines, all_flaw_lines, test_all_flaw_lines, test_all_total_locs, all_predicted_lines_number, x_percent=1)
    


# In[48]:


# Display Final Evaluation Results
top10acc = final_results_df[f'Top-{top_x} Accuracy'].tolist()[-2]
top_precision = final_results_df[f'Top-{top_x} Precision'].tolist()[-2]
top_recall = final_results_df[f'Top-{top_x} Recall'].tolist()[-2]
top_mrr = final_results_df[f'Reciprocal Rank-{top_x}'].tolist()[-2]
top_map = final_results_df[f'AP@{top_x}'].tolist()[-2]
top_mar = final_results_df[f'AR@{top_x}'].tolist()[-2]
ifa = final_results_df["IFA"].tolist()[-1]
print(f"Top-{top_x} Accuracy: {top10acc}")
print(f"Top-{top_x} Precision: {top_precision}")
print(f"Top-{top_x} Recall: {top_recall}")
print(f"Top-{top_x} Reciprocal Rank: {top_mrr}")
print(f"Top-{top_x} MAP: {top_map}")
print(f"Top-{top_x} MAR: {top_mar}")
print(f"Median IFA: {ifa}")
print(f"Effort@20%Recall: {effortXrecall}")
print(f"Recall@1%LOC: {recallXloc}")


# In[49]:


# Display Final Evaluation Results in Percentages
print("Accuracy results:")
print(f"Top-{top_x} Accuracy: {round(top10acc * 100, 1)}%")
print(f"Top-{top_x} Precision: {round(top_precision * 100, 1)}%")
print(f"Top-{top_x} Recall: {round(top_recall * 100, 1)}%")
print(f"Top-{top_x} MRR: {round(top_mrr * 100, 1)}%")
print(f"Top-{top_x} MAP: {round(top_map * 100, 1)}%")
print(f"Top-{top_x} MAR: {round(top_mar * 100, 1)}%")
print("\n")
print("Cost-effectiveness results:")
print(f"Median IFA: {round(ifa, 1)}")
print(f"Effort@20%Recall: {round(effortXrecall * 100, 1)}%")
print(f"Recall@1%LOC: {round(recallXloc * 100, 1)}%")


# In[ ]:




