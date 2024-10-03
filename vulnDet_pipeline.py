#!/usr/bin/env python
# coding: utf-8

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


# In[3]:


# Read dataset
root_path = os.getcwd()
dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))


# In[4]:


# Model checkpoint and fine-tuning logic
FINE_TUNE = False  # Set this to False if you don't want to fine-tune the model and load from checkpoint

checkpoint_dir = './checkpoints'
save_path = os.path.join(checkpoint_dir, 'best_weights.pt')


# In[5]:


# define functions
def save_checkpoint(filename, epoch, model, optimizer, scheduler, train_loss_per_epoch, val_loss_per_epoch, train_f1_per_epoch, val_f1_per_epoch):
    # If model is wrapped in DataParallel, save the underlying model's state_dict
    model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
    
    state = {
        'epoch': epoch,
        'model': model_state_dict,  # Use the correct state_dict
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loss_per_epoch': train_loss_per_epoch,
        'val_loss_per_epoch': val_loss_per_epoch,
        'train_f1_per_epoch': train_f1_per_epoch,
        'val_f1_per_epoch': val_f1_per_epoch
    }
    torch.save(state, filename)

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


# In[6]:


# Pre-trained tokenizer
model_variation = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True) #Tokenizer
#bert-base-uncased #bert-base # roberta-base # distilbert-base-uncased #distilbert-base # microsoft/codebert-base-mlm
# 'albert-base-v2'

# tokenizer = RobertaTokenizer(vocab_file="../../tokenizer_training/cpp_tokenizer/cpp_tokenizer-vocab.json",
#                              merges_file="../../tokenizer_training/cpp_tokenizer/cpp_tokenizer-merges.txt")


# In[7]:


# data split
val_ratio = 0.1
num_of_ratio = int(val_ratio * len(dataset))
data = dataset.iloc[0:-num_of_ratio, :]
test_data = dataset.iloc[-num_of_ratio:, :]
train_data = data.iloc[0:-num_of_ratio, :]
val_data = data.iloc[-num_of_ratio:, :]

# Shuffle dataset

train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
logger.info(f"Train data head: {train_data.head()}")
logger.info(f"Length of training data: {len(train_data)}")


#train_data = train_data[train_data["project"] != "Chrome"]
#logger.info(f"Length of training data without Chromium: {len(train_data)}")


train_data = train_data[["processed_func", "target", "flaw_line", "flaw_line_index"]]
train_data.head()


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


#val_data = val_data[val_data["project"] != "Chrome"]

val_data = pd.DataFrame(({'Text': val_data['processed_func'], 'Labels': val_data['target'], 'Lines':val_data['flaw_line'], 'Line_Index':val_data['flaw_line_index']}))
val_data.head()


#test_data = test_data[test_data["project"] != "Chrome"]

test_data = pd.DataFrame(({'Text': test_data['processed_func'], 'Labels': test_data['target'], 'Lines':test_data['flaw_line'], 'Line_Index':test_data['flaw_line_index']}))

logger.info(f"Train data length: {len(train_data)}")
logger.info(f"Validation data length: {len(val_data)}")
logger.info(f"Test data length: {len(test_data)}")


# In[8]:


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


# In[9]:


# Pre-trained model

model = AutoModelForSequenceClassification.from_pretrained(model_variation, num_labels=n_categories)
# Resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)


# Compute maximum length

X = tokenizer(
        text=X_train.tolist(),
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors='pt',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True
    )

max_len = getMaxLen(X)

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


# In[10]:


# Hyper-parameters

n_epochs = 10
lr = 2e-5 #5e-05
batch_size = 8 #16
patience = 5

optimizer = AdamW(model.parameters(),
                  lr = lr, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )


# In[11]:


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


# In[12]:


# # we do not retrain our pre-trained BERT and train only the last linear dense layer
# for param in model.roberta.parameters():
#     param.requires_grad = False


# In[13]:


if not FINE_TUNE and os.path.exists(save_path):
    pass
else:
    logger.info(f"Fine-tuning model: {model_variation}")
    # Train model
    
    # Initialize values for implementing Callbacks
    ## Early Stopping
    best_val_f1 = -1
    best_epoch = -1
    no_improvement_counter = 0
    ## Save best - optimal checkpointing
    #checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    #save_path = os.path.join(checkpoint_dir, 'best_weights.pt')
    
    print("Training...")
    milli_sec1 = int(round(time.time() * 1000))

    logger.info("Starting training...")
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    train_f1_per_epoch = []
    val_f1_per_epoch = []
    
    for epoch_num in range(n_epochs):
        logger.info(f'Epoch: {epoch_num + 1}')
    
        #Training
        model.train()
        train_loss = 0
        total_preds = []
        total_labels = []
        for step_num, batch_data in enumerate(tqdm(train_dataloader, desc='Training')):
    
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
    
            # clear previously calculated gradients
            model.zero_grad() # optimizer.zero_grad()
    
            # get model predictions for the current batch
            output = model(input_ids = input_ids, attention_mask=att_mask) # , labels=labels
    
            # compute the loss between actual and predicted values
            loss = loss_fun(output.logits, labels) #loss = output.loss #output[0]
            # add on to the total loss
            train_loss += loss.item()
    
            # backward pass to calculate the gradients
            loss.backward()
    
            # clip the gradients to 1.0. It helps in preventing the exploding gradient problem
            clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
    
            # update parameters
            optimizer.step()
            scheduler.step()
    
            # Print training loss after each batch
            #print("Epoch {}/{} - Batch {}/{} - Training Loss: {:.4f}".format(epoch_num+1, n_epochs, step_num+1, len(train_dataloader), loss.item()))
    
            # model predictions are stored on GPU. So, push it to CPU
            preds = np.argmax(output.logits.cpu().detach().numpy(),axis=-1)
            # append the model predictions
            total_preds+=list(preds)
            total_labels+=labels.cpu().numpy().tolist()
    
        train_loss_per_epoch.append(train_loss / len(train_dataloader))
        train_accuracy=accuracy_score(total_labels, total_preds)
        if n_categories > 2:
            train_precision=precision_score(total_labels, total_preds, average='macro')
            train_recall=recall_score(total_labels, total_preds, average='macro')
            train_f1=f1_score(total_labels, total_preds, average='macro')
        else:
            train_precision=precision_score(total_labels, total_preds)
            train_recall=recall_score(total_labels, total_preds)
            train_f1=f1_score(total_labels, total_preds)
            train_roc_auc=roc_auc_score(total_labels, total_preds)
        train_f2 = (5*train_precision*train_recall) / (4*train_precision+train_recall)
    
        #Validation
        model.eval()
        valid_loss = 0
        valid_pred = []
        actual_labels = []
        with torch.no_grad():
            for step_num_e, batch_data in enumerate(tqdm(val_dataloader, desc='Validation')):
                input_ids, att_mask, labels = [data.to(device) for data in batch_data]
    
                output = model(input_ids = input_ids, attention_mask=att_mask) # , labels=labels
    
                preds = np.argmax(output.logits.cpu().detach().numpy(), axis=-1)
                valid_pred+=list(preds)
                actual_labels+=labels.cpu().numpy().tolist()
    
                loss = loss_fun(output.logits, labels) #loss = output.loss #output[0]
                valid_loss += loss.item()
    
        val_loss_per_epoch.append(valid_loss / len(val_dataloader))
        val_accuracy=accuracy_score(actual_labels, valid_pred)
        if n_categories > 2:
            val_precision=precision_score(actual_labels, valid_pred, average='macro')
            val_recall=recall_score(actual_labels, valid_pred, average='macro')
            val_f1=f1_score(actual_labels, valid_pred, average='macro')
        else:
            val_precision=precision_score(actual_labels, valid_pred)
            val_recall=recall_score(actual_labels, valid_pred)
            val_f1=f1_score(actual_labels, valid_pred)
            val_roc_auc=roc_auc_score(actual_labels, valid_pred)
        val_f2 = (5*val_precision*val_recall) / (4*val_precision+val_recall)
    
        #print("Epoch {}/{} - Train Loss: {:.4f} - Valid Loss: {:.4f}".format(epoch_num+1, n_epochs, train_loss_per_epoch[-1], val_loss_per_epoch[-1]))
        #print("Epoch {}/{} - Train F1: {:.4f} - Valid F1: {:.4f}".format(epoch_num+1, n_epochs, train_f1, val_f1))
        logger.info(f"Epoch {epoch_num + 1}/{n_epochs} - Train Loss: {train_loss_per_epoch[-1]:.4f} - Valid Loss: {val_loss_per_epoch[-1]:.4f}")
        logger.info(f"Epoch {epoch_num + 1}/{n_epochs} - Train F1: {train_f1:.4f} - Valid F1: {val_f1:.4f}")

    
        train_f1_per_epoch.append(train_f1)
        val_f1_per_epoch.append(val_f1)
    
        total_epochs = epoch_num + 1
        # Implement Callbacks: Early Stopping and save best
        # Check if the validation F1 score has improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch_num + 1
            no_improvement_counter = 0 # Reset the counter
    
            # Save the best model checkpoint
            save_checkpoint(save_path, epoch_num+1, model, optimizer.state_dict(), scheduler.state_dict(), train_loss_per_epoch, val_loss_per_epoch, train_f1_per_epoch, val_f1_per_epoch)
            logger.info(f"Model saved at epoch {epoch_num + 1}")
        else:
            no_improvement_counter += 1
    
            if no_improvement_counter >= patience:
                # print("No improvement for", patience, "consecutive epochs.")
                # print("Early stopping after epoch No.", total_epochs)
                # print("Best model after epoch No", best_epoch)
                # print("Best achieved val_f1 = ", best_val_f1)
                logger.info(f"Early stopping after epoch {total_epochs}. Best epoch: {best_epoch} with best F1 score: {best_val_f1:.4f}")
                break

    milli_sec2 = int(round(time.time() * 1000))
    print("Training is completed after", milli_sec2-milli_sec1)

    epochs = range(1, total_epochs + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss_per_epoch, label ='training loss')
    ax.plot(epochs, val_loss_per_epoch, label = 'validation loss' )
    ax.set_title('Training and Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    #plt.show()
    plt.savefig('losses.png')
    plt.close()
    
    
    epochs = range(1, total_epochs + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, train_f1_per_epoch, label = 'training F1-score')
    ax.plot(epochs, val_f1_per_epoch, label = 'validation F1-score')
    ax.set_title('Training and Validation F1-scores')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('F1-score')
    ax.legend()
    #plt.show()
    plt.savefig('f-scores.png')
    plt.close()


# In[14]:


# Load best model from checkpoint during training with early stopping

checkpoint = torch.load(save_path, map_location=device)
# If model is wrapped in DataParallel, load state_dict directly into the underlying model
if torch.cuda.device_count() > 1:
    model.module.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint['model'])
model.to(device)


# Make predictions on the testing set
logger.info("Starting testing...")
test_start_time = time.time()
model.eval()
test_pred = []
actual_labels = []
test_loss = 0
with torch.no_grad():
    for step_num, batch_data in enumerate(tqdm(test_dataloader, desc='Testing')):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]

        output = model(input_ids = input_ids, attention_mask=att_mask) #, labels= labels

        loss = loss_fun(output.logits, labels) #loss = output.loss #output[0]
        test_loss += loss.item()

        preds = np.argmax(output.logits.cpu().detach().numpy(), axis=-1)
        test_pred+=list(preds)
        actual_labels+=labels.cpu().numpy().tolist()


# In[15]:


# compute evaluation metrics
class_report = classification_report(actual_labels, test_pred)
logger.info(f"Classification Report:\n{class_report}")
test_end_time = time.time()
testing_time = test_end_time - test_start_time
print("Testing is completed after", testing_time)
print("Perception time is", int(testing_time/len(test_pred)))

total_test_loss = test_loss/len(test_dataloader)
accuracy=accuracy_score(actual_labels, test_pred)
if n_categories > 2:
    precision=precision_score(actual_labels, test_pred, average='macro')
    recall=recall_score(actual_labels, test_pred, average='macro')
    f1=f1_score(actual_labels, test_pred, average='macro')
else:
    precision=precision_score(actual_labels, test_pred)
    recall=recall_score(actual_labels, test_pred)
    f1=f1_score(actual_labels, test_pred)
    roc_auc=roc_auc_score(actual_labels, test_pred)
f2 = (5*precision*recall) / (4*precision+recall)

print("Accuracy:%.2f%%"%(accuracy*100))
print("Precision:%.2f%%"%(precision*100))
print("Recall:%.2f%%"%(recall*100))
print("F1 score:%.2f%%"%(f1*100))
print("F2 score:%.2f%%"%(f2*100))
if roc_auc:
    print("Roc_Auc score:%.2f%%"%(roc_auc*100))

conf_matrix = confusion_matrix(actual_labels, test_pred)
tn, fp, fn, tp = conf_matrix.ravel()
#acc = ((tp+tn)/(tp+tn+fp+fn))

print("TP=",tp)
print("TN=",tn)
print("FP=",fp)
print("FN=",fn)
#print(conf_matrix)
sn.heatmap(conf_matrix, annot=True)


# In[16]:


# Export classification report

method = "forSequence"


# Create the path
path = os.path.join(root_path, 'results', model_variation.split("/")[-1], method, str(seed))

# Create directory if it doesn't exist
os.makedirs(path, exist_ok=True)

# Define the CSV file path
csv_file_path = os.path.join(path, f"{seed}.csv")

# Write data to CSV
data = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "f2": f2,
    "roc_auc": roc_auc
}

# Write to CSV
with open(csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data.keys())
    writer.writeheader()
    writer.writerow(data)


# Compute the average values of the classication metrics considering the results for all different seeders

# Define a dictionary to store cumulative sum of metrics
cumulative_metrics = defaultdict(float)
count = 0  # Counter to keep track of number of CSV files

# Iterate over all CSV files in the results folder
results_folder = os.path.join(root_path, "results", model_variation.split("/")[-1], method)

for root, dirs, files in os.walk(results_folder):
    for filename in files:
        if filename.endswith(".csv") and filename != "avg.csv":
            csv_file_path = os.path.join(root, filename)

            with open(csv_file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    for metric, value in row.items():
                        cumulative_metrics[metric] += float(value)
            count += 1

# Compute average values
average_metrics = {metric: total / count for metric, total in cumulative_metrics.items()}

# Print average values
print(average_metrics)

# Define the path for the average CSV file
avg_csv_file_path = os.path.join(root_path, "results", model_variation.split("/")[-1], method, "avg.csv")

# Write average metrics to CSV
with open(avg_csv_file_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=average_metrics.keys())
    writer.writeheader()
    writer.writerow(average_metrics)

# # Clean up
# del model
# torch.cuda.empty_cache()


# In[313]:


import lime
import shap
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
from captum.attr import DeepLiftShap


# In[408]:


EXPLAINER = "ATTENTION"  # or "LIME" or "DEEPLIFTSHAP" or "ATTENTION" based on user choice
logger.info(f"Initializing {EXPLAINER} explainer for Positive predictions...")

EXPLAIN_ONLY_TP = True


# In[409]:


if EXPLAIN_ONLY_TP:
    # Identify True Positives (where the predicted label and actual label are both 1)
    true_positive_indices = [i for i, (pred, label) in enumerate(zip(test_pred, Y_test.tolist())) if pred == 1 and label == 1]
    logger.info(f"Selected {len(true_positive_indices)} True Positives for explanations.")
    positive_indices = true_positive_indices
else:
    # Identify True Positives and False Positives
    positive_indices = [i for i, pred in enumerate(test_pred) if pred == 1]  # Indexes of Positive predictions (TPs + FPs)
    logger.info(f"Generating explanations for {len(positive_indices)} Positive predictions (TPs and FPs)...")


# In[410]:


# Function to predict probabilities for LIME
def predict_proba_func_lime(texts):
    model.eval()
    encodings = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_len
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits.cpu().numpy()
        
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    return probabilities


# In[411]:


# Function to initialize the explainer (LIME or SHAP)
def initialize_explainer():
    if EXPLAINER == "LIME":
        model.to(device)
        return LimeTextExplainer(class_names=['Non-Vulnerable', 'Vulnerable'], random_state=seed)
    elif EXPLAINER == "DEEPLIFTSHAP":
        model.to(device)
        return DeepLiftShap(model)  # Initialize DeepLiftShap with the model
    else:
        raise ValueError(f"Unknown explainer: {EXPLAINER}")

# Initialize the explainer
if EXPLAINER == "LIME" or EXPLAINER == "DEEPLIFTSHAP":
    explainer = initialize_explainer()


# In[412]:


# Function to tokenize the function into lines and tokens
def tokenize_function_to_lines_and_tokens(function_code):
    # Split function into lines based on newline characters
    lines = function_code.split('\n')
    
    # Tokenize each line
    tokenized_lines = []
    for line in lines:
        tokens = tokenizer.tokenize(line)
        tokenized_lines.append(tokens)
    
    return lines, tokenized_lines

# Function to compute LIME values for each line by summing the token-level values
def compute_lime_values_per_line(tokenized_lines, token_scores_dict):
    line_lime_scores = []

    # Iterate over tokenized lines
    for tokens in tokenized_lines:
        line_score = 0  # Initialize line score
        for token in tokens:
            # Retrieve the LIME score for the token if it exists
            if token in token_scores_dict:
                line_score += token_scores_dict[token]
        
        # Store the summed LIME score for the line
        line_lime_scores.append(line_score)
    
    return line_lime_scores

# Function to compute DeepLiftSHAP values for each line by summing the token-level values
def compute_deepliftshap_values_per_line(tokenized_lines, token_attributions):
    line_deepliftshap_scores = []

    token_idx = 0
    for tokens in tokenized_lines:
        line_score = sum(token_attributions[token_idx:token_idx+len(tokens)])
        line_deepliftshap_scores.append(line_score)
        token_idx += len(tokens)
    
    return line_deepliftshap_scores

# Function to compute attention values for each line
def compute_attention_values_per_line(tokenized_lines, attention_scores):
    line_scores = []
    token_idx = 0  # Keeps track of the token index
    
    for tokens in tokenized_lines:
        line_score = sum(attention_scores[token_idx:token_idx+len(tokens)])
        line_scores.append(line_score)
        token_idx += len(tokens)
    
    return line_scores

# Function to clean special token values (<s>, </s>, padding)
def clean_special_token_values(all_values, padding=True):
    # Special token in the beginning of the sequence
    all_values[0] = 0
    if padding:
        # Set the last non-zero value (representing the </s> token) to zero
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        # Special token at the end of the sequence
        all_values[-1] = 0
    return all_values


# In[413]:


positive_samples = [test_data['Text'].tolist()[i] for i in positive_indices]  # Extract Positive samples from test data

# Initialize a list to store the LIME explanations
explanation_results = []
# Loop through all positive samples (True Positives and False Positives)
for i, sample in enumerate(positive_samples):

    if EXPLAINER == "LIME":
        # Print logs every 10 samples
        if (i + 1) % 10 == 0:
            logger.info(f"Generating LIME explanation for sample {i + 1}/{len(positive_samples)}")
        
        # Generate explanation using the LIME explainer
        explanation = explainer.explain_instance(
            sample,  # The text/code snippet to explain
            predict_proba_func_lime,  # The function to predict probabilities
            num_features=20,  # Number of features to include in the explanation
            num_samples = 50,
            labels=[1]  # Target class (Vulnerable)
        )

    elif EXPLAINER == "DEEPLIFTSHAP":
        # Print logs every 10 samples
        if (i + 1) % 10 == 0:
            logger.info(f"Generating DEEPLIFTSHAP explanation for sample {i + 1}/{len(positive_samples)}")
        
        # Generate explanation using the SHAP explainer
        # Tokenize the function into lines and tokens
        lines, tokenized_lines = tokenize_function_to_lines_and_tokens(sample)

        # Encode the sample (input) and get embeddings
        encodings = tokenizer(sample, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)

        # Get the embeddings for input ids using the model's embedding layer
        input_embeddings = model.get_input_embeddings()(encodings['input_ids'])

        # Compute DeepLiftSHAP values per token
        num_baselines = 16
        baseline_inputs = torch.zeros_like(input_embeddings).repeat((num_baselines, 1, 1)).to(device)

        attributions = explainer.attribute(inputs=input_embeddings, baselines=baseline_inputs)

        # Convert attributions to a list of token-level attributions
        token_attributions = attributions.squeeze().tolist()

        # Clean attributions for special tokens (e.g., <s>, </s>, padding)
        token_attributions = clean_special_token_values(token_attributions, padding=True)

        # Compute DeepLiftSHAP values per line
        line_deepliftshap_scores = compute_deepliftshap_values_per_line(tokenized_lines, token_attributions)

        # Create a list of tuples containing (line_index, line_text, deepliftshap_score)
        explanation = [(line_idx, line, line_deepliftshap_scores[line_idx]) for line_idx, line in enumerate(lines)]

    
    elif EXPLAINER == "ATTENTION":
        # Print logs every 10 samples
        if (i + 1) % 10 == 0:
            logger.info(f"Generating ATTENTION-based explanation for sample {i + 1}/{len(positive_samples)}")
        
        # Tokenize the function into lines and tokens
        lines, tokenized_lines = tokenize_function_to_lines_and_tokens(sample)
        
        # Get model predictions along with attention weights
        with torch.no_grad():
            encodings = tokenizer(sample, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
            outputs = model(**encodings, output_attentions=True)
            logits = outputs.logits.cpu().detach().numpy()
            attentions = outputs.attentions  # Attention weights from each layer 

        batch_attention = attentions[0][0]

        # Summarize across heads by averaging the attention scores
        attention_summary = torch.mean(batch_attention, dim=0).cpu().numpy()  # Average across heads

        # Sum the attention each token receives from others (this gives a score for each token)
        token_attention_scores = np.sum(attention_summary, axis=0)  # Shape: (sequence_length,)

         # Clean attention scores for special tokens (e.g., <s>, </s>, padding)
        token_attention_scores = clean_special_token_values(token_attention_scores, padding=True)

        # Compute attention values per line
        line_attention_scores = compute_attention_values_per_line(tokenized_lines, token_attention_scores)

        # Create a list of tuples containing (line_index, line_text, attention_score)
        explanation = [(line_idx, line, line_attention_scores[line_idx]) for line_idx, line in enumerate(lines)]
    
    explanation_results.append(explanation)
    


# In[414]:


all_ranked_lines = []
for idx, explanation in enumerate(explanation_results):
    if idx % 10 == 0:
        logger.info(f"Explanation for Positive Sample {idx + 1}:")

    if EXPLAINER == "LIME":
        token_scores = explanation.as_list()
    elif EXPLAINER == "DEEPLIFTSHAP":
        token_scores = explanation

    if EXPLAINER == "ATTENTION":
      token_scores_dict = {line_idx: score for line_idx, line, score in explanation}
    elif EXPLAINER == "LIME":
        token_scores_dict = {}
        for token, score in token_scores:
            token_scores_dict[token] = score
    elif EXPLAINER == "DEEPLIFTSHAP":
        token_scores_dict = {line_idx: score for line_idx, line, score in token_scores}

    # Get the corresponding function code
    function_code = positive_samples[idx]
    
    # Tokenize the function into lines and tokens
    lines, tokenized_lines = tokenize_function_to_lines_and_tokens(function_code)
    
    # Compute values for each line
    if EXPLAINER == "DEEPLIFTSHAP":
        line_scores = compute_deepliftshap_values_per_line(tokenized_lines, token_scores_dict)
    elif EXPLAINER == "LIME":
        line_scores = compute_lime_values_per_line(tokenized_lines, token_scores_dict)
    elif EXPLAINER == "ATTENTION":
        line_scores = [token_scores_dict.get(line_idx, 0) for line_idx in range(len(lines))]
    
    # Create a list of tuples containing (line_index, line_text, lime_score)
    line_scores_with_text = [(line_idx, line, line_scores[line_idx]) for line_idx, line in enumerate(lines)]
    
    # Sort the lines by score in descending order
    ranked_lines = sorted(line_scores_with_text, key=lambda x: x[2], reverse=True)
    all_ranked_lines.append(ranked_lines)
    
    # Print the ranked lines
    if idx % 10 == 0:  # Log every 10th sample
        logger.info(f"Ranked lines for Positive Sample {idx + 1}:")
        for line_idx, line_text, score in ranked_lines[:3]:  # Only print top 3 lines
            logger.info(f"Line {line_idx}: {line_text} (Score: {score})")

    # Optionally, show the explanation in a notebook
    if EXPLAINER == "LIME":
        explanation.show_in_notebook(text=True)


# In[415]:


# Helper function to parse flaw lines
def parse_flaw_lines(flaw_line_str):
    """
    Parse flaw_line string into a list of integers.
    
    :param flaw_line_str: A string of comma-separated line numbers (e.g., '36,37,40').
    :return: List of integers representing the flaw lines.
    """
    if pd.isna(flaw_line_str) or flaw_line_str == '':
        return []
    else:
        return [int(x) for x in flaw_line_str.split(',')]

# Function to compute Top-X Accuracy for each function
def compute_top_x_accuracy(ranked_lines, flaw_lines, top_x=10):
    """
    Compute Top-X Accuracy: Measures whether at least one actual vulnerable line appears in the top-X ranking.
    
    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices (integers).
    :param top_x: The number of top lines to consider (default is 10).
    :return: 1 if at least one vulnerable line is in the top-X, else 0.
    """
    top_x_lines = ranked_lines[:top_x]  # Get the top-X ranked lines
    return 1 if any(line_index in flaw_lines for line_index, _, _ in top_x_lines) else 0

# Function to evaluate Top-X Accuracy across all functions
def evaluate_top_x_accuracy_for_all(all_ranked_lines, all_flaw_lines, top_x=10):
    """
    Evaluate Top-X Accuracy for all functions.
    
    :param all_ranked_lines: List of ranked lines for all functions.
    :param all_flaw_lines: List of actual vulnerable line indices (comma-separated strings) for all functions.
    :param top_x: The number of top lines to consider for Top-X Accuracy (default is 10).
    :return: Top-X Accuracy as a percentage of functions with at least one vulnerable line in the top X ranked lines.
    """
    successes = 0
    total_functions = len(all_ranked_lines)
    
    for ranked_lines, flaw_line_str in zip(all_ranked_lines, all_flaw_lines):
        flaw_lines = parse_flaw_lines(flaw_line_str)
        successes += compute_top_x_accuracy(ranked_lines, flaw_lines, top_x)
    
    # Return the percentage of functions where at least one vulnerable line was found in the top-X
    return (successes / total_functions) * 100

# Function to compute Initial False Alarm (IFA)
def compute_ifa(ranked_lines, flaw_lines):
    """
    Compute Initial False Alarm (IFA): Counts how many false alarms (non-vulnerable lines) occur before the first vulnerable line.
    
    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices.
    :return: Number of false alarms until the first vulnerable line is found.
    """
    ifa = 0
    for line_index, _, _ in ranked_lines:
        if line_index not in flaw_lines:
            ifa += 1
        else:
            break  # Stop counting when the first vulnerable line is found
    return ifa

# Function to compute Effort@X%Recall
def compute_effort_at_x_percent_recall(ranked_lines, flaw_lines, total_loc, x_percent=20):
    """
    Compute Effort@X%Recall: Measures the amount of effort (in LOC) to find X% of the actual vulnerable lines.
    
    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices.
    :param total_loc: Total number of lines of code (LOC) in the function.
    :param x_percent: Percentage of vulnerable lines to find for Effort@X%Recall (default is 20%).
    :return: The effort (as a ratio of inspected LOC to total LOC) required to find X% of the vulnerable lines.
    """
    total_vulnerable_lines = len(flaw_lines)
    if total_vulnerable_lines == 0:
        return 1.0  # If no vulnerable lines, maximum effort (full LOC inspected)
    
    # Calculate the number of vulnerable lines we need to find (X% of the total vulnerable lines)
    target_vulnerable_lines = max(1, int((x_percent / 100) * total_vulnerable_lines))
    
    # Iterate over ranked lines to count how much effort (LOC) is spent to find X% of the vulnerable lines
    inspected_lines = 0
    found_vulnerable_lines = 0
    
    for line_index, _, _ in ranked_lines:
        inspected_lines += 1
        if line_index in flaw_lines:
            found_vulnerable_lines += 1
        
        # Stop when we find X% of vulnerable lines
        if found_vulnerable_lines >= target_vulnerable_lines:
            break
    
    # Effort is the ratio of inspected lines (LOC) to the total number of lines (LOC)
    return inspected_lines / total_loc

# Function to compute Recall@1%LOC
def compute_recall_at_x_percent_loc(ranked_lines, flaw_lines, total_loc, x_percent=1):
    """
    Compute Recall@X%LOC: Measures the proportion of actual vulnerable lines that can be found in the top X% of LOC.
    
    :param ranked_lines: List of tuples (line_index, line_text, score) sorted by score.
    :param flaw_lines: List of actual vulnerable line indices.
    :param total_loc: Total number of lines of code (LOC) in the function.
    :param x_percent: Percentage of LOC to consider for the recall (default is 1%).
    :return: The recall for the top X% LOC.
    """
    # Calculate how many lines correspond to X% of the total LOC
    top_x_percent_loc = max(1, int((x_percent / 100) * total_loc))

    # Count how many vulnerable lines are found within the top X% LOC
    found_vulnerable_lines = 0
    inspected_lines = 0
    for line_index, _, _ in ranked_lines[:top_x_percent_loc]:
        inspected_lines += 1
        if line_index in flaw_lines:
            found_vulnerable_lines += 1
        if inspected_lines >= top_x_percent_loc:
            break
    
    # Recall is the ratio of correctly located vulnerable lines to the total number of actual vulnerable lines
    return found_vulnerable_lines / len(flaw_lines) if flaw_lines else 0.0


# In[416]:


# Function to evaluate all metrics for each function
def evaluate_vulnerability_detection(all_ranked_lines, all_flaw_lines, all_total_locs, top_x=10, effort_percent=20, loc_percent=1):
    """
    Evaluate the XAI methods using Top-X Accuracy, IFA, Effort@X%Recall, Recall@X%LOC for all functions.

    :param all_ranked_lines: List of ranked lines for all functions.
    :param all_flaw_lines: List of actual vulnerable line indices for all functions.
    :param all_total_locs: List of total number of lines of code (LOC) for all functions.
    :param top_x: Number of top-ranked lines to consider for Top-X Accuracy.
    :param effort_percent: Percentage of vulnerable lines to find for Effort@X%Recall.
    :param loc_percent: Percentage of LOC to consider for Recall@X%LOC.
    :return: DataFrame with individual and average results for each function.
    """
    results = []
    
    for i, ranked_lines in enumerate(all_ranked_lines):
        flaw_line_index = all_flaw_lines[i]
        total_loc = all_total_locs[i]

        # Even if there are no flaw lines, we still compute line-level evaluation for false positives
        flaw_lines = parse_flaw_lines(flaw_line_index) if pd.notna(flaw_line_index) else []

        # Compute each metric
        top_x_accuracy = compute_top_x_accuracy(ranked_lines, flaw_lines, top_x)
        ifa = compute_ifa(ranked_lines, flaw_lines)
        effort_at_x_percent_recall = compute_effort_at_x_percent_recall(ranked_lines, flaw_lines, total_loc, effort_percent)
        recall_at_x_percent_loc = compute_recall_at_x_percent_loc(ranked_lines, flaw_lines, total_loc, loc_percent)

        result = {
            f'Top-{top_x} Accuracy': top_x_accuracy,
            'IFA': ifa,
            f'Effort@{effort_percent}%Recall': effort_at_x_percent_recall,
            f'Recall@{loc_percent}%LOC': recall_at_x_percent_loc
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


# In[417]:


all_flaw_lines = [test_data['Line_Index'].tolist()[i] for i in positive_indices]  # Extract the flaw line indexes for each positive sample
all_flaw_lines_text = [test_data['Lines'].tolist()[i] for i in positive_indices]  # Extract the flaw lines for each positive sample
all_total_locs = [len(test_data['Text'].tolist()[i].split('\n')) for i in positive_indices]  # Compute total LOC for each positive sample

# Example usage:
final_results_df = evaluate_vulnerability_detection(all_ranked_lines, all_flaw_lines, all_total_locs, top_x=10, effort_percent=20, loc_percent=1)

# Display results
print(final_results_df)


# # In[ ]:
#
#
#
#
#
# # In[418]:
#
#
# number = 753
# sample = positive_samples[number]
# print(sample)
#
# lines, tokenized_lines = tokenize_function_to_lines_and_tokens(sample)
# print(lines)
# print(tokenized_lines)
#
# print(explanation_results[number])
#
# token_scores_dict = {line_idx: score for line_idx, line, score in explanation_results[number]}
# print(token_scores_dict)
#
# line_scores = [token_scores_dict.get(line_idx, 0) for line_idx in range(len(lines))]
# print(line_scores)
#
# line_scores_with_text = [(line_idx, line, line_scores[line_idx]) for line_idx, line in enumerate(lines)]
# print(line_scores_with_text)
#
# ranked_lines = sorted(line_scores_with_text, key=lambda x: x[2], reverse=True)
# print(ranked_lines)
#
# all_total_locs[number]
#
# print(all_flaw_lines[number])
# flaw_indexes = [int(x) for x in all_flaw_lines[number].split(',')]
# print(flaw_indexes)
#
# # find the text of the all_flaw_lines[0] indexes
# print(all_flaw_lines_text[number])
#
# print(lines[flaw_indexes[0]])
#
# # check ifa for this sample
# count = 0
# for rank_line in ranked_lines:
#     rank_line_index = rank_line[0]
#
#     if rank_line_index in flaw_indexes:
#         break
#     else:
#         count = count + 1
# print("IFA = ", count)
    

