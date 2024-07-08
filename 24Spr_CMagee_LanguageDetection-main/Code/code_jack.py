#### Data exploration

# ----------------- Imports ----------------- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets import load_dataset, concatenate_datasets

# ----------------- Loading Data ----------------- #
# Loading the english dataset
ds = load_dataset("mozilla-foundation/common_voice_16_1", "en", streaming=True)
# print(ds.head())

# Langugage dicitonary (from carrie's code)
LANGUAGES = {'en':"English", 'ca': 'Catalan', 'rw': 'Kinyarwanda','be': 'Belarusian', 'eo': 'Esperanto', 
             'de': 'German', 'fr': 'French', 'kab': 'Kabyle','es': 'Spanish', 'lg': 'Luganda', 'sw': 'Swahili',
             'fa': 'Persian','it': 'Italian', 'mhr': 'Meadow Mari',  'zh-CN': 'Chinese (China)', 'ba': 'Bashkir', 
             'ta': 'Tamil', 'ru': 'Russian','eu': 'Basque', 'th': 'Thai', 'pt': 'Portuguese',
             'pl': 'Polish','ug': 'Uyghur', 'lv': 'Latvian', 'ka': 'Georgian','ja': 'Japanese',
              'cy': 'Welsh', 'tr': 'Turkish','ckb': 'Central Kurdish', 'zh-HK': 'Chinese (Hong Kong)', 
              'nl': 'Dutch', 'uk': 'Ukrainian','uz': 'Uzbek', 'ar': 'Arabic','hu': 'Hungarian','zh-TW': 'Chinese (Taiwan)', 
              'cs': 'Czech', 'bn': 'Bengali', 'fy-NL': 'Frisian', 'kmr': 'Kurmanji Kurdish', 'ur': 'Urdu', 'gl': 'Galician',
              'yue': 'Cantonese'}

num_languages = len(LANGUAGES)
print("Number of Languages", num_languages)

#cv_16 = load_dataset("mozilla-foundation/common_voice_16_1", split="train")
cv_16_total = []
language_sizes = pd.DataFrame({"Language": [], "Number of Rows": []})

for lang_config, lang_name in LANGUAGES.items():
    try:
        # Appending Dataset
        dataset_path = f"mozilla-foundation/common_voice_16_1"
        dataset = load_dataset(dataset_path, lang_config, split="train")
        cv_16_total.append(dataset)
        print(f"Loaded {lang_name} dataset. \n")
        
        # Getting dataset specs
        # print(dataset.head(5))
        # print(f"{lang_name} shape: {dataset.shape} \n \n")
        # print(f"{lang_name} info: {dataset.info()} \n \n")
        
        # Language Spec Dataset
        current_row = {"Language": lang_name, "Number of Rows": dataset.shape[0]}
        language_sizes = language_sizes.append(current_row, ignore_index=True)
        
    except Exception as error:
        print(f"Error loading {lang_name} dataset: {error}")

cv_16 = concatenate_datasets(cv_16_total)

print()
print("Number of rows for each language in dataset:")
print(language_sizes)

# Plotting
language_sizes = language_sizes.sort_values(by="Number of Rows", ascending=False)
plt.bar(language_sizes["Languages"], language_sizes["Number of Rows"])
plt.xlabel("Language")
plt.ylabel("Number of Recordings")
plt.title("Frequency of Voice Recordings by Language")

## Trying to load a smaller portion of the dataset
train_english = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train", streaming=True)
indices = list(range(10))

selected_rows = []
for i, example in enumerate(train_english):
    if i in indices:
        selected_rows.append(example)
        
print(selected_rows)
for row in selected_rows:
    print(row)