#%%
import datasets
from datasets import load_dataset, concatenate_datasets, Audio, load_from_disk, Dataset
import pandas as pd
import numpy as np
import csv

import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    pipeline,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainerCallback)

#%%
############## FOR UPLOADING TO HF #######################
from huggingface_hub import login
access_token = "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"
login(token = access_token)

#%%
############## USING GPU ###########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#%%
############## LOADING & SPLITTING DATASET #######################
# dataset = load_from_disk("/home/ubuntu/github/24Spr_CMagee_LanguageDetection/Code/test_data_subset")
dataset = load_from_disk("/home/ubuntu/github/24Spr_CMagee_LanguageDetection/Code/test_data_subset")

dataset = dataset.shuffle(seed=22)
# dataset = dataset.select(range(300))
dataset = dataset.train_test_split(seed=42, shuffle=True, test_size=0.25)
print(dataset)

#%%
dataset = dataset.remove_columns(["client_id","path", "sentence", "up_votes", "down_votes","age","gender","accent","segment","variant"])
#%%
print(dataset["train"][0])
#%%
################### CREATING LABEL DICT ############################

langs = ['en', 'ca', 'rw', 'be', 'eo', 'de', 'fr', 'kab', 'es', 'lg', 'sw', 'fa', 'it', 'mhr', 'zh-CN', 'ba', 'ta', 'ru', 'eu', 'th', 'pt', 'pl', 'ja']

label2id, id2label = dict(), dict()
for i, label in enumerate(langs):
    label2id[label] = str(i)
    id2label[str(i)] = label
#%%
################ LOADING WAV2VEC2 FEATURE EXACTOR TO PROCESS AUDIO SIGNAL #########
model_id = "facebook/wav2vec2-base"
# using a feature extractor (not a tokenizer) to preprocess audio data for ASR 
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
        max_length=int(15.0*sampling_rate), 
        truncation=True)
    return inputs

data = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, num_proc=2) #added brackets and num_proc
dataset = data.rename_column("locale", "label")

#%%
############### CREATING COMPUTE METRICS FUNCTION #############
# creating a function that passes the predictions and true labels to compute function to calculate accuracy and f1
def compute_metrics(eval_pred):
    # Getting predictions and labels
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    # Getting scores
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    epoch = trainer.state.epoch  # assuming you have access to the current epoch number
    with open(f"Code/final_predictions/epoch_{epoch}_predictions_labels.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Prediction", "Label"])
        for pred, label in zip(predictions, labels):
            writer.writerow([pred, label])
    
    return {"accuracy": accuracy, "f1": f1}

def get_final_predictions(trainer, eval_dataset):
    
    eval_predicitons = trainer.predict(eval_dataset)
    
    predictions = eval_predicitons.predictions.argmax(axis=1)
    labels = eval_predicitons.label_ids
    
    accuracy = accuracy_score(labels, predictions)
    print(accuracy)
    
    return predictions, labels

class PushToHubCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Push the model checkpoint to the Hugging Face Hub
        trainer.push_to_hub()
    
#%%
# Splitting test set into test and validation
# Establishing ratios
validation_ratio = 0.6
test_ratio = 1 - validation_ratio

# Getting number of samples
num_samples = len(dataset["test"])
num_validation_samples = int(num_samples * validation_ratio)

# Splitting test dataset
validation_dataset = dataset["test"].select(range(num_validation_samples))
test_dataset = dataset["test"].select(range(num_validation_samples, num_samples))
dataset = {
    "train": dataset["train"],
    "validation": validation_dataset,
    "test": test_dataset
}
print(dataset)

# %%
# Assinging datasets
def map_to_device(example):
    example['label'] = example['label'].to(device)
    example['input_values'] = example['input_values'].to(device)
    
    return example

# Train
train_dataset = dataset["train"]
train_dataset = train_dataset.class_encode_column("label")

# Test
test_dataset = dataset["test"]
test_dataset = test_dataset.class_encode_column("label")

# Validation
validation_dataset = dataset["validation"]
validation_dataset = validation_dataset.class_encode_column("label")
#%%
####### LOAD MODEL FOR TRAINING #############
num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(model_id, num_labels=num_labels, label2id=label2id, id2label=id2label)
# model.to(device)

#The model needs to understand how to map its output predictions (which are numeric IDs) back to their corresponding label names. 
#It's a common practice to provide these mappings when fine-tuning a pretrained model for specific classification tasks in the Transformers library (not just for wav2vec2)

#%%
###### DEFINING TRAINING HYPERPARAMETERS ############
training_args = TrainingArguments(
    output_dir="wav2vec2-attempt2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    gradient_checkpointing=True,
    learning_rate=3e-4, 
    per_device_train_batch_size=9, #can change to 10 or to 16
    gradient_accumulation_steps=12, #can change to 8 or to 5
    per_device_eval_batch_size=9, #can change to 10
    num_train_epochs=6, #can chance to 10 or to 16 
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
    eval_dataset=validation_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[PushToHubCallback()]
)

trainer.train()

# Save the model and tokenizer to a directory
# trainer.save_pretrained("wav2vec2/model")
# feature_extractor.save_pretrained("wav2vec2/tokenizer")

trainer.push_to_hub("wav2vec2-attempt2")
feature_extractor.push_to_hub("feature_extractor")

print('------------GETTING METRICS-----------------')
# Metrics on Validation Dataset
validation_results = trainer.evaluate(eval_dataset=validation_dataset)
print(f"Validation Accuracy: {validation_results['eval_accuracy']}")
print(f"Validation F1 Score: {validation_results['eval_f1']}")

# Metrics on Test Dataset
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test Accuracy: {test_results['eval_accuracy']}")
print(f"Test F1 Score: {test_results['eval_f1']}")

# Metrics on Train Dataset
train_results = trainer.evaluate(eval_dataset=train_dataset)
print(f"Train Accuracy: {train_results['eval_accuracy']}")
print(f"Train F1 Score: {train_results['eval_f1']}")
# %%
#trainer.push_to_hub(repo_name = "WAV2VEC2_CAPSTONE_MODEL")

# Getting predictions and labels
train_predictions, train_labels = get_final_predictions(trainer, train_dataset)
validation_predictions, validation_labels = get_final_predictions(trainer, validation_dataset)
test_predictions, test_labels = get_final_predictions(trainer, test_dataset)


train_df = pd.DataFrame({
    "predictions": train_predictions,
    "labels": train_labels
})
validation_df = pd.DataFrame({
    "predictions": validation_predictions,
    "labels": validation_labels
})
test_df = pd.DataFrame({
    "predictions": test_predictions,
    "labels": test_labels
})

validation_df.to_csv("Code/final_predictions/validation_predictions.csv")
test_df.to_csv("Code/final_predictions/test_df.csv")
train_df.to_csv("Code/final_predictions/train_df.csv")
# %%
