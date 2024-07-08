### Doing EDA on the English Dataset

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import librosa

from datasets import load_dataset

# %%
# Loading Data - English training split
data = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", streaming=True)
# data.save_to_disk("common-voice-en-train")
dataloader = DataLoader(data, batch_size=32)

print(next(iter(data)))
# %%
# Demographic information
male_count = 0
female_count = 0
other_count = 0
for row in data:
    # print(row["gender"])
    if row['gender'] == 'male':
        male_count += 1
    elif row['gender'] == 'female':
        female_count += 1
    else:
        other_count += 1
print('-*50', '\n')
print(f"Number of males: {male_count}, Number of females: {female_count}, Other: {other_count}")
# Number of males: 534634, Number of females: 216130, Other: 339297]