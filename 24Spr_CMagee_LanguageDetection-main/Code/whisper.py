#%%
import datasets
from datasets import load_dataset, concatenate_datasets, Audio, load_from_disk
import torch
from transformers import WhisperFeatureExtractor
#%%
torch.cuda.empty_cache()

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#%%
from huggingface_hub import login
access_token_write = "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"
login(token = access_token_write)

#%%
### PREPARE DATASET
data = load_dataset("mageec/test_data_subset", split="train")
languages_to_include = ['en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su', 'yue', 'my', 'ca', 'nl', 'ht', 'lb', 'ps', 'pa', 'ro', 'ro', 'si', 'es', 'zh']
# Filter the dataset to include only rows with the specified languages
data = data.filter(lambda example: example["locale"] in languages_to_include, num_proc=4)
data = data.shuffle()
data = data.train_test_split(seed=42, shuffle=True, test_size=0.10)
#%%
### LOAD FEATURE EXTRACTOR
from transformers import AutoFeatureExtractor
model_id = "openai/whisper-tiny"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
sampling_rate = feature_extractor.sampling_rate
#%%
from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language=None, task="transcribe")

#%%
#%%
input_str = data["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

#%%
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language=None, task="transcribe")
# %%
### MAKING SAMPLING RATE SAME FOR ALL SAMPLES
data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))
#%%
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0] #computing input features (Log-Mel represent the spectral characteristics of an audio signal)
  
    language = batch["locale"]
    tokenizer.set_prefix_tokens(language=language) #creating prefix tokens to identify specific language
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids #tokenizes and encodes target text 
    return batch

#%%
data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=4)
torch.cuda.empty_cache()
#%%
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
#%%
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
#%%
import evaluate
wer_metric = evaluate.load("wer")
accuracy_metric = evaluate.load("accuracy")
#%%
def compute_metrics(pred):
    pred_ids = pred.predictions
    #print("Pred_Ids 0", pred_ids[0])
    #print("Pred_Ids 1029", pred_ids[1029])
    label_ids = pred.label_ids
    #print("Label_Ids 0", label_ids[0])
    #print("Label_Ids 1029", label_ids[1029])


    predictions_first = pred_ids[:, 0]
    predictions_first = predictions_first.astype(int)
    #print("Preds first", predictions_first[:3])
    labels_first = label_ids[:, 0]
    labels_first = labels_first.astype(int)
    #print("Labels first",labels_first[:3])
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #print(pred_str[0])
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    #print(label_str[0])

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    accuracy = accuracy_metric.compute(references=labels_first, predictions=predictions_first) 
    
    return {"wer": wer}

#%% 
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-05-01",  # change to a repo name of your choice
    per_device_train_batch_size=14,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-4,
    warmup_steps=350, #500
    max_steps=3000, #5000
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000, #1000
    eval_steps=1000, #1000
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer.train()

#%%
trainer.push_to_hub("whisper-05-01")
tokenizer.push_to_hub("whisper-05-01") 
