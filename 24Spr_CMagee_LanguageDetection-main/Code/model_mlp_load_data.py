####### Model Architecture Practice #########
# %%
####### Imports #######
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
# from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, load_from_disk, Audio
import soundfile
import librosa

from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# %%
###### Hyperparamters ######
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# Manualling setting to cpu while running on local
# device = torch.device('cpu')
print(f"Device being used: {device}")
print(torch.version.cuda)

n_epoch = 10
BATCH_SIZE = 1
LR = 0.0001
THRESHOLD = 0.5
SAVE_MODEL = True

###### Functions #######

# Defining Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
    
        return x
    
# Creating dataset class

class audioDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        X = self.features[index]
        y = self.targets[index]
        
        return X, y
        
def define_model(input_size, hidden_size, output_size):
    """
    
    """
    model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion

def extract_array(batch):
    audio_arrays = [x['array'] for x in batch['audio']]

def train_model(train_ds, input_size, hidden_size=100, output_size=1, epochs=n_epoch):
    """
 
    """
    model, optimizer, critierion = define_model(input_size, hidden_size, output_size)
    
    model = model.to(device)
    count = 0
    
    print(model)
    
    for epoch in range(epochs):
        
        model.train()
        
        train_loss, steps_train = 0, 0
        losses = []
        
        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch+1)) as pbar:
            
            for xdata, xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()
                
                output = model(xdata)
                
                # # Checking output ang target shapes
                # print(f"Target shape: {xtarget.shape}")
                # print(f"Output shape: {output.shape}")
                
                loss = critierion(output.float(), xtarget.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                count += 1
                steps_train += 1
                
                losses.append(loss.item())
                
                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))
    
    return model
                
def test_model(model, test_ds):
    """
    
    """
    model.eval()
    
    predicted_probs = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_ds:
            data = data.to(device)

            output = model(data).argmax(dim=1).to("cpu")
            targ = target.argmax(dim=1)
            
            predicted_probs.append(output)
            labels.append(targ)
            
    # print(predicted_probs)
    # print(labels)

    # Use this for binary model
    # predictions = [1 if score >= THRESHOLD else 0 for score in predicted_probs]
    
    # For multi-class
    predictions = predicted_probs #.to("cpu")

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    # auc = roc_auc_score(labels, predictions)
    
    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1: {f1}")
    # print(f"Test AUC: {auc}")
    print(classification_report(labels, predictions))
    print(confusion_matrix(labels, predictions))
    
def find_percentile(data, percentile):
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = (percentile / 100) * (n + 1)
    if index.is_integer():
        percentile_value = sorted_data[int(index) - 1]
    else:
        lower_index = int(index) - 1
        upper_index = lower_index + 1
        percentile_value = (sorted_data[lower_index] + sorted_data[upper_index]) / 2
    return int(percentile_value)

# From Carrie
def label2id_fn(label):
    label_mapping = {'en': 0, 'ca': 1, 'rw': 2, 'be': 3, 'eo': 4, 'de': 5, 'fr': 6, 
                     'kab': 7, 'es': 8, 'lg': 9, 'sw': 10, 'fa': 11, 'it': 12, 'mhr': 13, 
                     'zh-CN': 14, 'ba': 15, 'ta': 16, 'ru': 17, 'eu': 18, 'th': 19, 'pt': 20, 'pl':21, 'ja':22}
    return label_mapping.get(label)


def id2label_fn(label_id):
    label_mapping = {0: 'en', 1: 'ca', 2: 'rw', 3: 'be', 4: 'eo', 5: 'de', 6: 'fr', 
                         7: 'kab', 8: 'es', 9: 'lg', 10: 'sw', 11: 'fa', 12: 'it', 13: 'mhr', 
                         14: 'zh-CN', 15: 'ba', 16: 'ta', 17: 'ru', 18: 'eu', 19: 'th', 20: 'pt', 21: 'pl', 22: 'ja'}
    return label_mapping.get(label_id)

def preprocess_function(row):
    audio_arrays = [x["array"] for x in row["audio"]]
    
    inputs = feature_extractor(
        audio_arrays, sampling_rate = feature_extractor.sampling_rate
    )
    
    return inputs

######## Training script ###########
# %%
if __name__ == '__main__':
    
    # Loading data
    data_path = "/home/ubuntu/github/24Spr_CMagee_LanguageDetection/Code/final_data_subset"
    data = load_from_disk(data_path)
    data = data.train_test_split(shuffle=True, seed=42, test_size=0.15)
    
    # Getting sampling rate
    model_id = "facebook/wav2vec2-base"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    sampling_rate = feature_extractor.sampling_rate
    
    # Applying sampling rate to all columns
    data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))
    print(data["train"][0])
    
    # Encoding data
    encoded_data = data.map(preprocess_function, remove_columns='audio', batched=True, num_proc=2)
    encoded_data = encoded_data.rename_column("locale", "label")
    print(encoded_data["train"][0].keys())
    
    # Splitting data
    train_dataset = encoded_data["train"]
    train_dataset = train_dataset.class_encode_column("label")
    test_dataset = encoded_data["test"]
    test_dataset = test_dataset.class_encode_column("label")
    print(train_dataset[0].keys())
    
    print("hello")
    print(train_dataset["input_values"])
    print("hello")
    train_arrays = [x["input_values"] for x in train_dataset]
    print("start train targets")
    train_targets = [x["label"] for x in train_dataset]
    
    test_arrays = [x["input_values"] for x in test_dataset]
    test_targets = [x["label"] for x in test_dataset]
    
    print("Lengths in Train:")
    print(set(len(array) for array in train_arrays))
    print("Lengths in Test:")
    print(set(len(array) for array in train_arrays))
    
    # Loading data
    train_dataset = audioDataset(train_arrays, train_targets)
    test_dataset = audioDataset(test_arrays, test_targets)
    
    train_loader = DataLoader(train_dataset, batch_sampler=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # # %%
    # Training data
    # model = train_model(train_loader, input_size=eighty, hidden_size=1100, output_size=3)
        
    # %%
    # Evaluating data
    # test_model(model, test_loader)
    # %%
