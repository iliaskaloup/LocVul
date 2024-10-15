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
from torch.nn.utils import clip_grad_norm_

from transformers import set_seed
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForSeq2SeqLM #, BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler

from evaluate import load

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
os.makedirs(checkpoint_dir, exist_ok=True)


# In[3]:


def save_checkpoint(filename, epoch, model, optimizer, scheduler, train_loss_per_epoch, val_loss_per_epoch, train_rouge_per_epoch, val_rouge_per_epoch):
    # If model is wrapped in DataParallel, save the underlying model's state_dict
    model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
    
    state = {
        'epoch': epoch,
        'model': model_state_dict,  # Use the correct state_dict
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loss_per_epoch': train_loss_per_epoch,
        'val_loss_per_epoch': val_loss_per_epoch,
        'train_rouge_per_epoch': train_rouge_per_epoch,
        'val_rouge_per_epoch': val_rouge_per_epoch
    }
    torch.save(state, filename)


# Data Processing

# In[4]:


# Read dataset
root_path = os.getcwd()
dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))
#dataset = dataset.iloc[0:1000,: ]
dataset = dataset.dropna(subset=["processed_func"])


# In[5]:


# data split
val_ratio = 0.1
num_of_ratio = int(val_ratio * len(dataset))
data = dataset.iloc[0:-num_of_ratio, :]
test_data = dataset.iloc[-num_of_ratio:, :]
train_data = data.iloc[0:-num_of_ratio, :]
val_data = data.iloc[-num_of_ratio:, :]


# In[6]:


# release some memory
del dataset


# In[7]:


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


# In[8]:


# Function to replace "/~/" with "\n" in the 'Lines' column
def replace_delimiter_with_newline(data):
    # Replace "/~/" with "\n" in the 'Lines' column
    data['Lines'] = data['Lines'].str.replace('/~/', '\n')
    return data

train_data = replace_delimiter_with_newline(train_data)
val_data = replace_delimiter_with_newline(val_data)
test_data = replace_delimiter_with_newline(test_data)


# In[9]:


test_data["Text"][1]


# In[10]:


test_data["Lines"][1]


# Tokenization

# In[11]:


model_variation = "Salesforce/codet5-base" # "google-t5/t5-base" # Salesforce/codet5-base"
#tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True)


# In[12]:


# Get the actual tokenized lengths
def getMaxLen(X):

    # Code for identifying max length of the data samples after tokenization using transformer tokenizer
    
    max_length = 0
    max_row = 0
    
    # Iterate over each sample in your dataset
    for i, input_ids in enumerate(X['input_ids']):
        # Convert input_ids to a PyTorch tensor
        input_ids_tensor = torch.tensor(input_ids)
        # Calculate the length of the tokenized sequence for the current sample
        length = torch.sum(input_ids_tensor != tokenizer.pad_token_id).item()
        # Update max_length and max_row if the current length is greater
        if length > max_length:
            max_length = length
            max_row = i

    logger.info(f"Max length of tokenized data: {max_length}")
    logger.info(f"Row with max length:: {max_row}")
    
    return max_length

target_encodings = tokenizer(
    text=train_data['Lines'].tolist(),
    add_special_tokens=True,
    truncation=False,  # Do not truncate
    padding=False      # No padding to see actual lengths
)

# Compute max length for the Lines column
max_len_lines = getMaxLen(target_encodings)

if max_len_lines > 512:
    max_len_lines = 512
else:
    max_len_lines = max_len_lines
logger.info(f"Maximum tokenized length of Lines: {max_len_lines}")
max_len_lines = 128


# In[13]:


def tokenize_data(data, max_len_lines):
    input_encodings = tokenizer(
        data['Text'].tolist(),
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        add_special_tokens=True
    )
    
    target_encodings = tokenizer(
        data['Lines'].tolist(),
        max_length=max_len_lines,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        add_special_tokens=True
    )

    input_encodings['labels'] = target_encodings['input_ids']
    
    return input_encodings

# Tokenize train, validation, and test data
train_encodings = tokenize_data(train_data, max_len_lines)
val_encodings = tokenize_data(val_data, max_len_lines)
test_encodings = tokenize_data(test_data, max_len_lines)


# Prepare DataLoaders

# In[14]:


# Define batch size
batch_size = 8


# In[15]:


# Create TensorDatasets
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_encodings['labels'])
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_encodings['labels'])
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_encodings['labels'])

# Create DataLoaders
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)


# Model Initialization

# In[16]:


# Load the CodeT5 model
model = AutoModelForSeq2SeqLM.from_pretrained(model_variation)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device {device}")

print(model.to(device))
print("No. of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)


# Training Loop

# In[17]:


# Hyper-parameters
learning_rate = 5e-5
num_epochs = 10
patience = 5  # Early stopping patience

optimizer = AdamW(model.parameters(), lr=learning_rate, eps = 1e-8)
max_steps = len(train_loader) * num_epochs
lr_scheduler = get_scheduler(
    name='linear', optimizer=optimizer, num_warmup_steps=max_steps // 5, num_training_steps=max_steps
)


# In[18]:


## Training Loop
# Early Stopping and Checkpointing Setup
rouge_metric = load("rouge")
best_val_rouge = -1
best_epoch = -1
no_improvement_counter = 0

# Initialize lists for tracking loss and ROUGE scores
train_loss_per_epoch = []
val_loss_per_epoch = []
train_rouge_per_epoch = []
val_rouge_per_epoch = []

# Start Training
milli_sec1 = int(round(time.time() * 1000))
logger.info("Starting training...")

for epoch_num in range(num_epochs):
    logger.info(f'Epoch: {epoch_num + 1}')
    
    # Training
    model.train()
    train_loss = 0
    total_preds = []
    total_labels = []

    for step_num, batch_data in enumerate(tqdm(train_loader, desc='Training')):
        input_ids, attention_mask, labels = [data.to(device) for data in batch_data]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Clip gradients to prevent exploding gradients
        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        
        # Collect predictions and actual labels for ROUGE
        preds = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        total_preds.extend(decoded_preds)
        total_labels.extend(decoded_labels)

    # Compute average training loss
    train_loss_per_epoch.append(train_loss / len(train_loader))

    # Compute ROUGE for training set
    train_rouge_scores = rouge_metric.compute(predictions=total_preds, references=total_labels, use_stemmer=True)
    # Check if ROUGE score is a scalar (float) or a detailed dictionary
    if isinstance(train_rouge_scores["rougeL"], dict):
        avg_train_rouge = train_rouge_scores["rougeL"].mid.fmeasure #* 100
    else:
        avg_train_rouge = train_rouge_scores["rougeL"] #* 100  # For scalar case
    train_rouge_per_epoch.append(avg_train_rouge)

    # Validation
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for step_num_e, batch_data in enumerate(tqdm(val_loader, desc='Validation')):
            input_ids, attention_mask, labels = [data.to(device) for data in batch_data]

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            
            # Collect predictions and actual labels for ROUGE
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_len_lines)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            val_preds.extend(decoded_preds)
            val_labels.extend(decoded_labels)

    # Compute average validation loss
    val_loss_per_epoch.append(val_loss / len(val_loader))

    # Compute ROUGE for validation set
    val_rouge_scores = rouge_metric.compute(predictions=val_preds, references=val_labels, use_stemmer=True)
    # Check if ROUGE score is a scalar (float) or a detailed dictionary
    if isinstance(val_rouge_scores["rougeL"], dict):
        avg_val_rouge = val_rouge_scores["rougeL"].mid.fmeasure #* 100
    else:
        avg_val_rouge = val_rouge_scores["rougeL"] #* 100  # For scalar case
    val_rouge_per_epoch.append(avg_val_rouge)
    
    logger.info(f"Epoch {epoch_num + 1}/{num_epochs} - Train Loss: {train_loss_per_epoch[-1]:.4f} - Valid Loss: {val_loss_per_epoch[-1]:.4f}")
    logger.info(f"Epoch {epoch_num + 1}/{num_epochs} - Train ROUGE-L: {avg_train_rouge:.4f} - Valid ROUGE-L: {avg_val_rouge:.4f}")

    # Implement Early Stopping and Save Best Model
    if avg_val_rouge > best_val_rouge:
        best_val_rouge = avg_val_rouge
        best_epoch = epoch_num + 1
        no_improvement_counter = 0

        # Save the best model
        #torch.save(model.state_dict(), save_path)
        save_checkpoint(save_path, epoch_num+1, model, optimizer.state_dict(), lr_scheduler.state_dict(), train_loss_per_epoch, val_loss_per_epoch, train_rouge_per_epoch, val_rouge_per_epoch)
        logger.info(f"Model saved at epoch {epoch_num + 1} with ROUGE-L: {best_val_rouge:.4f}")
    else:
        no_improvement_counter += 1

        if no_improvement_counter >= patience:
            logger.info(f"Early stopping after {epoch_num + 1} epochs. Best epoch: {best_epoch} with ROUGE-L: {best_val_rouge:.4f}")
            break

# Training Complete
milli_sec2 = int(round(time.time() * 1000))
logger.info(f"Training completed in {(milli_sec2 - milli_sec1) // 1000} seconds.")

# Plotting Loss and ROUGE Scores
epochs = range(1, len(train_loss_per_epoch) + 1)

# Loss plot
plt.figure()
plt.plot(epochs, train_loss_per_epoch, label='Training Loss')
plt.plot(epochs, val_loss_per_epoch, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses.png')
plt.close()

# ROUGE-L plot
plt.figure()
plt.plot(epochs, train_rouge_per_epoch, label='Training ROUGE-L')
plt.plot(epochs, val_rouge_per_epoch, label='Validation ROUGE-L')
plt.title('Training and Validation ROUGE-L Scores')
plt.xlabel('Epochs')
plt.ylabel('ROUGE-L')
plt.legend()
plt.savefig('rouge_scores.png')
plt.close()


# Evaluation

# In[19]:


# Load best model from checkpoint during training with early stopping

checkpoint = torch.load(save_path, map_location=device)
# If model is wrapped in DataParallel, load state_dict directly into the underlying model
if torch.cuda.device_count() > 1:
    model.module.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint['model'])
model.to(device)


# In[20]:


# Make predictions on the testing set
logger.info("Starting testing...")
test_start_time = time.time()

model.eval()
test_preds = []
actual_labels = []
test_loss = 0

with torch.no_grad():
    for step_num, batch_data in enumerate(tqdm(test_loader, desc='Testing')):
        input_ids, attention_mask, labels = [data.to(device) for data in batch_data]

        # Generate predictions
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_len_lines)

        # Decode predicted sequences and actual labels
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        test_preds.extend(decoded_preds)
        actual_labels.extend(decoded_labels)

test_end_time = time.time()
testing_time = test_end_time - test_start_time

# Display the total testing time and average time per sample
print("Testing completed after", testing_time)
print("Perception time per sample:", int(testing_time / len(test_preds)))


# In[21]:


def extract_rouge_value(rouge_scores, rouge_key):
    score = rouge_scores[rouge_key]
    if isinstance(score, dict):  # If it's a dictionary, extract the 'mid' fmeasure
        return score['mid'].fmeasure #* 100
    else:  # If it's a float, return the value directly
        return score #* 100
        
# Compute evaluation metrics using ROUGE
rouge_metric = load("rouge")
# Calculate ROUGE scores for the predictions and actual sequences
rouge_scores = rouge_metric.compute(predictions=test_preds, references=actual_labels, use_stemmer=True)
logger.info(f"ROUGE scores on test data: {rouge_scores}")

# Display detailed scores for ROUGE-1, ROUGE-2, and ROUGE-L
rouge1_score = extract_rouge_value(rouge_scores, 'rouge1')
rouge2_score = extract_rouge_value(rouge_scores, 'rouge2')
rougeL_score = extract_rouge_value(rouge_scores, 'rougeL')
rougeLsum_score = extract_rouge_value(rouge_scores, 'rougeLsum')
# Log the extracted ROUGE scores
logger.info(f"ROUGE-1: {rouge1_score:.4f}")
logger.info(f"ROUGE-2: {rouge2_score:.4f}")
logger.info(f"ROUGE-L: {rougeL_score:.4f}")
logger.info(f"ROUGE-Lsum: {rougeLsum_score:.4f}")


# In[22]:


# Optionally, save the predictions and true labels for further analysis
with open('test_predictions.txt', 'w') as f_pred, open('test_actual_labels.txt', 'w') as f_labels:
    for pred, label in zip(test_preds, actual_labels):
        f_pred.write(pred + "\n")
        f_labels.write(label + "\n")


# Generating Vulnerable Lines (Inference)

# In[23]:


# Function to generate vulnerable lines for a code snippet
def generate_vulnerable_lines(model, tokenizer, code_snippet, max_length):
    # Tokenize the input code snippet
    inputs = tokenizer(
        code_snippet,
        return_tensors='pt',
        truncation=True,
        padding='max_length',  # You can adjust padding as needed
        max_length=512  # The max length for the input code snippet
    ).to(device)

    # Generate predicted vulnerable lines using the model
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,  # Maximum length for the generated sequence (vulnerable lines)
        num_beams=4,  # Beam search for better results (you can adjust or remove for greedy search)
        early_stopping=True  # Stop generating once the model reaches an end token
    )

    vulnerable_lines = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return vulnerable_lines

model.eval()  # Set the model to evaluation mode
# Example usage with a code snippet
code_snippet = """void testFunction() {
    int x = 5;
    int y = x / 0; // Division by zero
    char data[10];
    strcpy(data, input); // Buffer overflow
    delete[] data; // Potential double free
}"""

# Generate vulnerable lines for the example code snippet
predicted_vulnerable_lines = generate_vulnerable_lines(model, tokenizer, code_snippet, max_len_lines)

# Output the result
print("Predicted Vulnerable Lines:")
print(predicted_vulnerable_lines)


# In[24]:


# Function to generate vulnerable lines for a code snippet
def generate_vulnerable_lines(model, tokenizer, code_snippet, max_length=128):
    # Tokenize the input code snippet
    inputs = tokenizer(
        code_snippet,
        return_tensors='pt',
        truncation=True,
        padding='max_length',  # You can adjust padding as needed
        max_length=512  # The max length for the input code snippet
    ).to(device)

    # Generate predicted vulnerable lines using the model
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length,  # Maximum length for the generated sequence (vulnerable lines)
        num_beams=4,  # Beam search for better results (you can adjust or remove for greedy search)
        early_stopping=True  # Stop generating once the model reaches an end token
    )

    # Decode the predicted tokens into text
    vulnerable_lines = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return vulnerable_lines

# Set the model to evaluation mode
model.eval()

# Assuming your test dataset is loaded into a DataFrame called 'test_data'
# Use the first sample from the test set (column 'processed_func' contains the source code)
first_code_snippet = test_data['Text'].iloc[0]

# Generate vulnerable lines for the first sample
predicted_vulnerable_lines = generate_vulnerable_lines(model, tokenizer, first_code_snippet)

# Output the result
print("First Test Sample Code Snippet:")
print(first_code_snippet)


# In[25]:


print("\nPredicted Vulnerable Lines:")
print(predicted_vulnerable_lines)


# In[ ]:




