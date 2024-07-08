#%%
import datasets
from datasets import load_dataset, concatenate_datasets, Audio, load_from_disk
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
    AutoModelForAudioClassification)

torch.cuda.empty_cache()
#%%
############## FOR UPLOADING TO HF #######################
from huggingface_hub import login
access_token_write = "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"
login(token = access_token_write)

#%%
############## USING GPU ###########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#%%
############## LOADING & SPLITTING DATASET #######################
dataset = load_from_disk("/home/ubuntu/24Spr_CMagee_LanguageDetection/Code/test_data_subset")
dataset = dataset.train_test_split(seed=42, shuffle=True, test_size=0.10)

#%%
dataset = dataset.remove_columns(["client_id","path", "sentence", "up_votes", "down_votes","age","gender","accent","segment","variant"])
#%%
################### CREATING LABEL DICT ############################
# TO MAKE IT EASIER FOR THE MODEL TO GET THE LABEL NAME FROM THE LABEL ID, WE CREATE A DICT
# THAT MAKES THE LABEL NAME TO AN INT AND VICE VERSA 
langs = ['en', 'ca', 'rw', 'be', 'eo', 'de', 'fr', 'kab', 'es', 'lg', 'sw', 'fa', 'it', 'mhr', 'zh-CN', 'ba', 'ta', 'ru', 'eu', 'th', 'pt', 'pl', 'ja']
#all_langs = sorted(list(set(langs)))
#id2label = {idx: all_langs[idx] for idx in range(len(all_langs))}
#label2id = {v: k for k, v in id2label.items()}
label2id, id2label = dict(), dict()
for i, label in enumerate(langs):
    label2id[label] = str(i)
    id2label[str(i)] = label
#%%
################ LOADING WAV2VEC2 FEATURE EXACTOR TO PROCESS AUDIO SIGNAL #########
model_id = "facebook/wav2vec2-base"
#using a feature extractor (not a tokenizer) to preprocess audio data for ASR 
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
sampling_rate = feature_extractor.sampling_rate
#%%
#################### RESAMPLING AUDIO DATA TO SUIT WAV2VEC2 MODEL ####################
dataset = dataset.cast_column("audio",Audio(sampling_rate=sampling_rate))
#%%
################## CREATING PREPROCESSING FUNCTION #################
# THE FUNCTION WILL...
  # call the audio column to load, and if necessary, resample the audio file
  # checks if the SR of the audio file matches the SR of the audio data the model (w2v in this case) was pretrained with 
  # set a max input length to batch longer inputs w/o truncating
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rate, 
        max_length=int(15.0*sampling_rate), #maybe change to 13 from 11 #originally was 15  
        truncation=True)
    return inputs

#%%

dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, num_proc=2) #added brackets and num_proc #
dataset = dataset.rename_column("locale", "label")

#%%
############### CREATING COMPUTE METRICS FUNCTION #############
# creating a function that passes the predictions and true labels to compute function to calculate accuracy
import evaluate
import numpy as np
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

#%%
############### CREATING COMPUTE METRICS FUNCTION #############
# creating a function that passes the predictions and true labels to compute function to calculate accuracy and f1
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids, average = "weighted")
    accuracy= accuracy['accuracy']
    f1 = f1['f1']
    return {"accuracy": accuracy, "f1 score": f1}

#%%
train_dataset = dataset["train"]
train_dataset = train_dataset.class_encode_column("label")
test_dataset = dataset["test"]
test_dataset = test_dataset.class_encode_column("label")

#%%
####### LOAD MODEL FOR TRAINING #############
num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(model_id, num_labels=num_labels, label2id=label2id, id2label=id2label)


#The model needs to understand how to map its output predictions (which are numeric IDs) back to their corresponding label names. 
#It's a common practice to provide these mappings when fine-tuning a pretrained model for specific classification tasks in the Transformers library (not just for wav2vec2)

#%%
###### DEFINING TRAINING HYPERPARAMETERS ############
training_args = TrainingArguments(
    output_dir="wave2vec2_capstone",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    gradient_checkpointing=True,
    learning_rate=3e-4, 
    per_device_train_batch_size=9, #change to 9
    gradient_accumulation_steps=12, #change to 12
    per_device_eval_batch_size=9, #change to 9
    num_train_epochs=8, #change to 12 
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    push_to_hub=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()
#%%
# Save processor and create model card
trainer.push_to_hub("wave2vec2_capstone")



#%%



