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
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import soundfile
import librosa

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

######## Training script ###########
# %%
if __name__ == '__main__':
    
    # Get training data
    training_languages = {'af': 'Afrikaans', 'vot': 'Votic', 'lo': "Lo-Toga"}
    data_list = []
    target_list = []
    
    for lang_config, lang_name in training_languages.items():
        try:
            # Getting features
            dataset_path = f"mozilla-foundation/common_voice_16_1"
            dataset = load_dataset(dataset_path, lang_config, split="train", streaming=False, use_auth_token=True)
            # dataset = load_dataset(dataset_path, lang_config, split="train",streaming=False)
            data_list.append(dataset)
            print(f"Loaded {lang_name} dataset.")
            
            # Getting targets
            lang_length = len(dataset)
            target_list += [lang_name] * lang_length
            
        except Exception as error:
            print(f"Error loading {lang_name} dataset: {error}")
    
    training_data = concatenate_datasets(data_list)
    training_data.remove_columns(["client_id", "up_votes", "down_votes", "segment", "variant"])
    print(training_data)
    # print('-'*100, "\n", target_list)
    
    # Getting audio arrays
    train_arrays = [x['audio']['array'] for x in training_data]
    # print(train_arrays)
    array_lengths = [len(array) for array in train_arrays]
    mean_length = np.mean(array_lengths)
    median_length = np.median(array_lengths)
    
    longest_array_train = len(max(train_arrays, key=len))
    print(f"LENGTH OF LONGEST ARRAY: {longest_array_train}")
    print(f"MEAN LENGTH: {mean_length}")
    print(f"MEDIAN LENGTH: {median_length}")
    
    # Plotting array lengths
    plt.hist(array_lengths, bins=30)
    plt.title("Length of Decoded Arrays - Train")
    plt.show()
    
    # Padding arrays
    # print(array_lengths)
    eighty = find_percentile(array_lengths, 80)
    print(f"EIGHTITH PERCENTILE: {eighty}")
    # padded_arrays = [np.pad(array, (0, (longest_array_train - len(array)) ), mode='constant', constant_values=0) for array in train_arrays]
    padded_arrays = []
    for array in train_arrays:
        if len(array) > eighty:
            padded_arrays.append(array[:eighty])
        else:
            padded_arrays.append(np.pad(array, (0, (eighty - len(array)) ), mode='constant', constant_values=0))
    
    print(set([len(array) for array in padded_arrays]))
    
    target_array = np.array(target_list).reshape(-1, 1)
    
    ohe = OneHotEncoder(sparse_output=False)
    train_targets = ohe.fit_transform(target_array)
    # print(train_targets)
    
    train_dataset = audioDataset(padded_arrays, train_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get testing data
    data_list = []
    target_list = []
    
    for lang_config, lang_name in training_languages.items():
        try:
            # Getting features
            dataset_path = f"mozilla-foundation/common_voice_16_1"
            dataset = load_dataset(dataset_path, lang_config, split="test",streaming=False, use_auth_token=True)
            data_list.append(dataset)
            print(f"Loaded {lang_name} dataset.")
            
            # Getting targets
            lang_length = len(dataset)
            target_list += [lang_name] * lang_length
            
        except Exception as error:
            print(f"Error loading {lang_name} dataset: {error}")
    
    testing_data = concatenate_datasets(data_list)
    testing_data.remove_columns(["client_id", "up_votes", "down_votes", "segment", "variant"])
    print(testing_data)
    
    # Getting audio arrays
    test_arrays = [x['audio']['array'] for x in testing_data]
    # print(test_arrays)
    array_lengths = [len(array) for array in test_arrays]
    mean_length = np.mean(array_lengths)
    median_length = np.median(array_lengths)
    
    longest_array = len(max(test_arrays, key=len))
    print(f"LENGTH OF LONGEST ARRAY: {longest_array}")
    print(f"MEAN LENGTH: {mean_length}")
    print(f"MEDIAN LENGTH: {median_length}")
    
    # Plotting array lengths
    plt.hist(array_lengths, bins=30)
    plt.title("Length of Decoded Arrays - Test")
    plt.show()
    
    # Padding arrays
    # padded_arrays = [np.pad(array, (0, (longest_array - len(array)) ), mode='constant', constant_values=0) for array in test_arrays]
    padded_arrays = []
    for array in test_arrays:
        if len(array) > eighty:
            padded_arrays.append(array[:eighty])
        else:
            padded_arrays.append(np.pad(array, (0, (eighty - len(array)) ), mode='constant', constant_values=0))
    
    # padded_arrays = []
    # for array in train_arrays
    
    print(set([len(array) for array in padded_arrays]))
    
    # test_targets = [0 if x == "Afrikaans" else 1 if x == "Votic" else 0 for x in target_list]
    target_array = np.array(target_list).reshape(-1, 1)
    test_targets = ohe.transform(target_array)
    
    test_dataset = audioDataset(padded_arrays, test_targets)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # %%
    # Training data
    model = train_model(train_loader, input_size=eighty, hidden_size=1100, output_size=3)
        
# %%
# Evaluating data
test_model(model, test_loader)
# %%
