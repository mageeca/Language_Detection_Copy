# %%
#### PREDICTING RESULTS FROM TRAINED MODEL ####
# Imports
import datasets
from datasets import load_dataset, Audio, load_from_disk

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    pipeline,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor)

# %%
# Loading Model
import requests

API_URL = "https://api-inference.huggingface.co/models/mageec/wave2vec2_capstone"
headers = {"Authorization": "Bearer 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"}

# def query(filename):
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers=headers, data=data)
#     return response.json()

# output = query("sample1.flac")

# %%
# Trying a different code
model_name = "mageec/wav2vec2-attempt2"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
# tokenizer = Wav2Vec2Processor.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

dataset = load_from_disk("/home/ubuntu/github/24Spr_CMagee_LanguageDetection/Code/test_data_subset")
dataset = dataset.shuffle(seed=22)
# dataset = dataset.select(range(300))
dataset = dataset.train_test_split(seed=42, shuffle=True, test_size=0.25)
dataset = dataset.rename_column("locale", "label")
print(dataset)

dataset_test = dataset["test"].select(range(100))

# %%
# predictions
def predict(examples, model):
    
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    labels = []
    probs = []
    
    for waveform in audio_arrays:
        input = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate
        )
    
        with torch.no_grad():
            logits = model(**input).logits
            
            predicted_class_ids = torch.argmax(logits).item()
            try:
                predicted_label = model.config.id2label[predicted_class_ids]
            except:
                predicted_label = "Unknown"
            labels.append(predicted_label)

            softmax_scores = torch.softmax(logits, dim=1)
            try:
                predicted_prob_score = softmax_scores[0][predicted_class_ids].item()
            except:
                predicted_prob_score = "Unknown"
            probs.append(predicted_prob_score)
            
    return labels, probs

labels, probs = predict(dataset_test, model)
labels, probs
# %%
# Loading dataset
dataset = load_from_disk("/home/ubuntu/github/24Spr_CMagee_LanguageDetection/Code/test_data_subset")
dataset = dataset.train_test_split(seed=42, shuffle=True, test_size=0.15)

dataset = dataset.remove_columns(["client_id","path", "sentence", "up_votes", "down_votes","age","gender","accent","segment","variant"])

sampling_rate = feature_extractor.sampling_rate
print(sampling_rate)

def preprocess_function(examples):
    
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    inputs = []
    
    for waveform in audio_arrays:
        input = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors='pt',
            max_length=sampling_rate, 
            truncation=True)
        
        inputs.append(input)
        
    return inputs

# data = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, num_proc=2) #added brackets and num_proc
dataset = dataset.rename_column("locale", "label")


inputs = preprocess_function(dataset["train"][:5])
print(len(inputs))
# inputs = feature_extractor(dataset["train"][3111]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt", padding=True)

## Carrie's Code
for input in inputs:
    with torch.no_grad():
        logits = model(**input).logits

    #getting the label class with the highest probability, and using model's id2label mapping to convert into actual label 
    import torch

    predicted_class_ids = torch.argmax(logits).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    predicted_label

    #%%
    # Getting the probability scores
    softmax_scores = torch.softmax(logits, dim=1)
    predicted_prob_score = softmax_scores[0][predicted_class_ids].item()
    actual_label = dataset[3111]['label']

    # Printing the predicted label along with its probability score
    print(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}, Probability Score: {predicted_prob_score}")
# %%

"""
# %%
# # Predicting
logits = []
for input in inputs:
    # print(input.shape)
    with torch.no_grad():
        logit = model(**input).logits
        print(logit.shape)
        logits.append(logit)

predicted_ids = []
for logit in logits:
    predicted_id = torch.argmax(logit).item()
    predicted_ids.append(predicted_id)
    
print(predicted_ids)

# labels = []
# for id in predicted_ids:
#     label = model.config.id2label[id]
#     labels.append(label)
# print(labels)

softmax_scores = torch.softmax(logits, dim=1)
predicted_prob_score = softmax_scores[0][predicted_ids].item()

print(softmax_scores, predicted_prob_score)

# print(f"Predicted logits: {logits}")
# probabilities = F.softmax(logits, dim=-1)
# predicted_ids = torch.argmax(probabilities, dim=-1)
# print(f"Predicted ids: {predicted_ids}")

# # Decode the predicted IDs to obtain the transcription
# transcription = tokenizer.batch_decode(predicted_ids)

# # Print the transcription
# print("Transcription:", transcription)
"""