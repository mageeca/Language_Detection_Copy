###### EDA OF FINAL DATA SUBSET #####

# %%
# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch
import datasets
from datasets import load_from_disk, concatenate_datasets, Audio, Dataset

# %%
# Loading data
dataset = load_from_disk("/home/ubuntu/github/24Spr_CMagee_LanguageDetection/Code/test_data_subset")
dataset

# %%
# Transforming to a pandas dataframe for easier analysis
data = dataset.to_pandas()
data.head()

# %%
audio_dicts = []
for row in dataset:
    audio_dicts.append(row["audio"])

audio_df = pd.DataFrame(audio_dicts)

# %% Narrowing Dataframe
cols = ["sentence", "age", "gender", "accent", "locale", "segment", "variant"]
data = data[cols]

audio_df = audio_df[["array", "sampling_rate"]]

data = pd.concat([data, audio_df], axis=1)
data.to_csv("data_dataframe.csv", index=False)
# %%
# Getting gender ratios
data = pd.read_csv("data_dataframe.csv")

gender = pd.DataFrame()

data.loc[data["gender"] == "", "gender"] = pd.NA

male_df = data.loc[data["gender"] == "male"]
female_df = data.loc[data["gender"] == "female"]

gender["group_count"] = data.groupby("locale").size()
gender["missing"] = data.groupby("locale").apply(lambda x: x["gender"].isnull().sum())
gender["missing_percent"] = gender["missing"] / gender["group_count"] * 100
gender["filled"] = data.loc[data["gender"].isna() == False].groupby("locale").size()
gender["female_count"] = female_df.groupby("locale").size()
gender["male_count"] = male_df.groupby("locale").size()
gender["female_percent"] = gender["female_count"] / gender["filled"] * 100
gender["male_percent"] = gender["male_count"] / gender["filled"] * 100

gender
# %%
# Plotting missing values
plt.bar(gender.index, gender["missing_percent"])
plt.title("Missing Gender Data by Language")
plt.xlabel("Language")
plt.xticks(rotation=45)
plt.ylabel("Percent")
plt.show()
# %%
gender[["female_percent", "male_percent"]].plot(kind='bar', figsize=(10,6), width=0.95)
plt.title("Gender Representation by Language")
plt.xlabel("Language")
plt.xticks(rotation=45)
plt.ylabel("Percent")
plt.show()
# %%
gender_full = data["gender"].value_counts()

plt.bar(gender_full.index, gender_full.values)
plt.title("Gender Ratios Across Full Dataset")
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.show()
# %%
# Getting Age Statistics
def fill_age(row):
    if row["age"] == "teens":
        return 18
    elif row['age'] == "twenties":
        return 25
    elif row["age"] == "thirties":
        return 35
    elif row["age"] == "fourties":
        return 45
    elif row["age"] == "fifties":
        return 55
    elif row["age"] == "sixties":
        return 65
    elif row["age"] == "seventies":
        return 75
    elif row["age"] == "eighties":
        return 85
    elif row["age"] == "nineties":
        return 95
    
data["age_number"] = data.apply(fill_age, axis=1)
data["age_number"].value_counts()
# %%
age_counts = data["age"].value_counts()

plt.bar(age_counts.index, age_counts.values)
plt.title("Age Distribution Across All Languages")
plt.xlabel("Age Bracket")
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.show()
# %%
age = pd.DataFrame()

age["counts"] = data.groupby("locale").size()
age["mean_age"] = data.groupby("locale")["age_number"].mean()
age["median_age"] = data.groupby("locale")["age_number"].median()
age["sd_age"] = data.groupby("locale")["age_number"].std()

age
# %%
plt.bar(age.index, age["mean_age"], yerr=age["sd_age"], capsize=5, alpha=0.7)
plt.title("Mean Age by Language")
plt.xlabel("Language")
plt.xticks(rotation=45)
plt.ylabel("Age (Years)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# %%
# Getting length of audio file
data["audio_length"] = data.apply(lambda x: len(x["array"]) / x["sampling_rate"], axis=1)
data.head()
# %%
data["array_length"] = data.apply(lambda x: len(x["array"]), axis=1)
data.head()
# %%
data["array_length"].value_counts()
# %%

# Get total length

# Get mean length of arrays