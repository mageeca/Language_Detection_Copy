#%%
from datasets import load_dataset, concatenate_datasets, Audio, load_from_disk, load_dataset
# %%
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
from huggingface_hub import login
access_token_write = "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"
login(token = access_token_write)
dataset = load_dataset("mageec/test_data_subset", split="train")
# %%
print(dataset["audio"])
# %%
from datasets import load_dataset
dataset = load_dataset("audiofolder", data_dir="/home/ubuntu/24Spr_CMagee_LanguageDetection/Code/test_data_subset")
# %%
print()