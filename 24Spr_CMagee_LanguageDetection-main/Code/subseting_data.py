#%%
import datasets
from datasets import load_dataset, concatenate_datasets, Audio, load_from_disk
#%%
import torch

if torch.cuda.is_available():
    print("CUDA (GPU) is available.")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
    print(f"CUDA capability: {torch.cuda.get_device_capability()}")
else:
    print("CUDA (GPU) is not available. Using CPU instead.")
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
#%%
from huggingface_hub import login
access_token_read = "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"
login(token = access_token_read)
#%%
#from huggingface_hub import notebook_login
#notebook_login()
#%%
#%%
english = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train").shuffle(seed=42)
english = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train[:5000]")
english.save_to_disk("data_subset1")
#%%
data1 = load_from_disk("data_subset1")
catalan = load_dataset("mozilla-foundation/common_voice_16_1", "ca", split="train").shuffle(seed=42)
catalan = load_dataset("mozilla-foundation/common_voice_16_1", "ca", split="train[:5000]")
data = concatenate_datasets([data1,catalan])
data.save_to_disk("data_subset2")
#%%
data2 = load_from_disk("data_subset2")
kinya = load_dataset("mozilla-foundation/common_voice_16_1", "rw", split="train").shuffle(seed=42)
kinya = load_dataset("mozilla-foundation/common_voice_16_1", "rw", split="train[:5000]")
data = concatenate_datasets([data2,kinya])
data.save_to_disk("data_subset3")
# %%
data3 = load_from_disk("data_subset3")
bela = load_dataset("mozilla-foundation/common_voice_16_1", "be", split="train").shuffle(seed=42)
bela = load_dataset("mozilla-foundation/common_voice_16_1", "be", split="train[:5000]")
data = concatenate_datasets([data3,bela])
data.save_to_disk("data_subset4")
# %%
data4 = load_from_disk("data_subset4")
esperanto = load_dataset("mozilla-foundation/common_voice_16_1", "eo", split="train").shuffle(seed=42)
esperanto = load_dataset("mozilla-foundation/common_voice_16_1", "eo", split="train[:5000]")
data = concatenate_datasets([data4, esperanto])
data.save_to_disk("data_subset5")
#%% 
data5 = load_from_disk("data_subset5")
german = load_dataset("mozilla-foundation/common_voice_16_1", "de", split="train").shuffle(seed=42)
german = load_dataset("mozilla-foundation/common_voice_16_1", "de", split="train[:5000]")
data = concatenate_datasets([data5, german])
data.save_to_disk("data_subset6")
#%%
data6 = load_from_disk("data_subset6")
french = load_dataset("mozilla-foundation/common_voice_16_1", "fr", split="train").shuffle(seed=42)
french = load_dataset("mozilla-foundation/common_voice_16_1", "fr", split="train[:5000]")
data = concatenate_datasets([data6, french])
data.save_to_disk("data_subset7")
# %%
data7 = load_from_disk("data_subset7")
kabyle = load_dataset("mozilla-foundation/common_voice_16_1", "kab", split="train").shuffle(seed=42)
kabyle = load_dataset("mozilla-foundation/common_voice_16_1", "kab", split="train[:5000]")
data = concatenate_datasets([data7, kabyle])
data.save_to_disk("data_subset8")
# %%
data8 = load_from_disk("data_subset8")
spanish = load_dataset("mozilla-foundation/common_voice_16_1", "es", split="train").shuffle(seed=42)
spanish = load_dataset("mozilla-foundation/common_voice_16_1", "es", split="train[:5000]")
data = concatenate_datasets([data8, spanish])
data.save_to_disk("data_subset9")

# %%
data9 = load_from_disk("data_subset9")
luganda = load_dataset("mozilla-foundation/common_voice_16_1", "lg", split="train").shuffle(seed=42)
luganda = load_dataset("mozilla-foundation/common_voice_16_1", "lg", split="train[:5000]")
data = concatenate_datasets([data9, luganda])
data.save_to_disk("data_subset10")
#%%
data10 = load_from_disk("data_subset10")
swahili = load_dataset("mozilla-foundation/common_voice_16_1", "sw", split="train").shuffle(seed=42)
swahili = load_dataset("mozilla-foundation/common_voice_16_1", "sw", split="train[:5000]")
data = concatenate_datasets([data10, swahili])
data.save_to_disk("data_subset11")
# %%
data11 = load_from_disk("data_subset11")
persian = load_dataset("mozilla-foundation/common_voice_16_1", "fa", split="train").shuffle(seed=42)
persian = load_dataset("mozilla-foundation/common_voice_16_1", "fa", split="train[:5000]")
data = concatenate_datasets([data11, persian])
data.save_to_disk("data_subset12")
#%%
data12 = load_from_disk("data_subset12")
italian = load_dataset("mozilla-foundation/common_voice_16_1", "it", split="train").shuffle(seed=42)
italian = load_dataset("mozilla-foundation/common_voice_16_1", "it", split="train[:5000]")
data = concatenate_datasets([data12, italian])
data.save_to_disk("data_subset13")
#%%
data13 = load_from_disk("data_subset13")
mm = load_dataset("mozilla-foundation/common_voice_16_1", "mhr", split="train").shuffle(seed=42)
mm = load_dataset("mozilla-foundation/common_voice_16_1", "mhr", split="train[:5000]")
data = concatenate_datasets([data13, mm])
data.save_to_disk("data_subset14")
#%%
data14 = load_from_disk("data_subset14")
chinese_china = load_dataset("mozilla-foundation/common_voice_16_1", "zh-CN", split="train").shuffle(seed=42)
chinese_china = load_dataset("mozilla-foundation/common_voice_16_1", "zh-CN", split="train[:5000]")
data = concatenate_datasets([data14, chinese_china])
data.save_to_disk("data_subset15")
#%%
data15 = load_from_disk("data_subset15")
bashkir = load_dataset("mozilla-foundation/common_voice_16_1", "ba", split="train").shuffle(seed=42)
bashkir = load_dataset("mozilla-foundation/common_voice_16_1", "ba", split="train[:5000]")
data = concatenate_datasets([data15, bashkir])
data.save_to_disk("data_subset16")
#%%
data16 = load_from_disk("data_subset16")
tamil = load_dataset("mozilla-foundation/common_voice_16_1", "ta", split="train").shuffle(seed=42)
tamil = load_dataset("mozilla-foundation/common_voice_16_1", "ta", split="train[:5000]")
data = concatenate_datasets([data16, tamil])
data.save_to_disk("data_subset17")
#%%
data17 = load_from_disk("data_subset17")
russian = load_dataset("mozilla-foundation/common_voice_16_1", "ru", split="train").shuffle(seed=42)
russian = load_dataset("mozilla-foundation/common_voice_16_1", "ru", split="train[:5000]")
data = concatenate_datasets([data17, russian])
data.save_to_disk("data_subset18")
#%%
data18 = load_from_disk("data_subset18")
basque = load_dataset("mozilla-foundation/common_voice_16_1", "eu", split="train").shuffle(seed=42)
basque = load_dataset("mozilla-foundation/common_voice_16_1", "eu", split="train[:5000]")
data = concatenate_datasets([data18, basque])
data.save_to_disk("data_subset19")
#%%
data19 = load_from_disk("data_subset19")
thai = load_dataset("mozilla-foundation/common_voice_16_1", "th", split="train").shuffle(seed=42)
thai = load_dataset("mozilla-foundation/common_voice_16_1", "th", split="train[:5000]")
data = concatenate_datasets([data19, thai])
data.save_to_disk("data_subset20")
#%%
data20 = load_from_disk("data_subset20")
portuguese = load_dataset("mozilla-foundation/common_voice_16_1", "pt", split="train").shuffle(seed=42)
portuguese = load_dataset("mozilla-foundation/common_voice_16_1", "pt", split="train[:5000]")
data = concatenate_datasets([data20, portuguese])
data.save_to_disk("data_subset21")

#%%
data21 = load_from_disk("data_subset21")
polish = load_dataset("mozilla-foundation/common_voice_16_1", "pl", split="train").shuffle(seed=42)
polish = load_dataset("mozilla-foundation/common_voice_16_1", "pl", split="train[:5000]")
data = concatenate_datasets([data21, polish])
data.save_to_disk("data_subset22")
# %%
data22 = load_from_disk("data_subset22")
japanese = load_dataset("mozilla-foundation/common_voice_16_1", "ja", split="train").shuffle(seed=42)
japanese = load_dataset("mozilla-foundation/common_voice_16_1", "ja", split="train[:5000]")
data = concatenate_datasets([data22, japanese])
data.save_to_disk("dataset")
# %%
data = load_from_disk("dataset")
# %%

