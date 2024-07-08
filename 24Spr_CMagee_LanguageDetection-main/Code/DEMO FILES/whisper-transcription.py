#%%
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor
from datasets import load_dataset, concatenate_datasets, Audio, load_from_disk
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from huggingface_hub import login
access_token_write = "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"
login(token = access_token_write)

# Load the trained model and tokenizer
model_id = "mageec/whisper-tiny-hi-capstone"
#tokenizer = WhisperTokenizer.from_pretrained("mageec/whisper-tiny-hi-capstone")
#processor = WhisperProcessor.from_pretrained("mageec/whisper-tiny-hi-capstone", language=None, task="transcribe")
# %%

# Load the test dataset
#test_dataset = load_from_disk("/home/ubuntu/24Spr_CMagee_LanguageDetection/Code/test_data_subset")

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES

from datasets import load_dataset
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

bos_token_id = processor.tokenizer.all_special_ids[-106]
decoder_input_ids = torch.tensor([[1, bos_token_id]])

dataset = load_dataset("mageec/test_data_subset", split="train", streaming=True)
dataset = dataset.shuffle()
#%%
sample = next(iter(dataset))["audio"]
#%%
#%%
input_features = processor(sample["array"], sampling_rate=16000, return_tensors="pt").input_features
print("Input features shape:", input_features.shape)
with torch.no_grad():
    logits = model(input_features, decoder_input_ids=decoder_input_ids).logits
    generated_ids = model.generate(input_features, max_length=16_000, use_cache=True)

transcriptions = processor.batch_decode(generated_ids , skip_special_tokens=True)
print(transcriptions)

#%%
