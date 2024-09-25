#!/usr/bin/env python
# coding: utf-8

# In[336]:


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


# In[337]:


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


# In[338]:


# Read dataset
root_path = os.getcwd()
dataset = pd.read_csv(os.path.join(root_path, 'data', 'dataset.csv'))


# In[339]:


# Model checkpoint and fine-tuning logic
FINE_TUNE = True  # Set this to False if you don't want to fine-tune the model and load from checkpoint

checkpoint_dir = './checkpoints'
save_path = os.path.join(checkpoint_dir, 'best_weights.pt')


# In[340]:


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


# In[341]:


# Pre-trained tokenizer
model_variation = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_variation, do_lower_case=True) #Tokenizer
#bert-base-uncased #bert-base # roberta-base # distilbert-base-uncased #distilbert-base # microsoft/codebert-base-mlm
# 'albert-base-v2'

# tokenizer = RobertaTokenizer(vocab_file="../../tokenizer_training/cpp_tokenizer/cpp_tokenizer-vocab.json",
#                              merges_file="../../tokenizer_training/cpp_tokenizer/cpp_tokenizer-merges.txt")


# In[343]:


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


train_data = train_data[train_data["project"] != "Chrome"]
logger.info(f"Length of training data without Chromium: {len(train_data)}")


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


val_data = val_data[val_data["project"] != "Chrome"]

val_data = pd.DataFrame(({'Text': val_data['processed_func'], 'Labels': val_data['target'], 'Lines':val_data['flaw_line'], 'Line_Index':val_data['flaw_line_index']}))
val_data.head()


test_data = test_data[test_data["project"] != "Chrome"]

test_data = pd.DataFrame(({'Text': test_data['processed_func'], 'Labels': test_data['target'], 'Lines':test_data['flaw_line'], 'Line_Index':test_data['flaw_line_index']}))

logger.info(f"Train data length: {len(train_data)}")
logger.info(f"Validation data length: {len(val_data)}")
logger.info(f"Test data length: {len(test_data)}")


# In[344]:


# Pre-processing step: Under-sampling

sampling = True
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
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Resize model embedding to match new tokenizer

model.resize_token_embeddings(len(tokenizer))


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
model.load_state_dict(checkpoint["model"])
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


# In[480]:


