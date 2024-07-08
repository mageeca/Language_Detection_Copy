#%%
import datasets
from datasets import load_dataset, Audio, load_from_disk

import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    pipeline,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
    AutoModelForAudioClassification)
#%%
#################### OPTION 1: USING A PIPELINE ######################
#loading dataset but might want to create a new one for inference
#dataset = load_from_disk("/home/ubuntu/24Spr_CMagee_LanguageDetection/Code/final_data_subset")
dataset = load_dataset("mozilla-foundation/common_voice_16_1", "ja", split="train") #change to common voice 
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
model_id = "mageec/wave2vec2_capstone"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
#%%
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rate, 
        max_length=sampling_rate, 
        truncation=True)
    return inputs

dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, num_proc=2) #added brackets and num_proc #
dataset = dataset.rename_column("locale", "label")
#%%
audio_file = dataset[3111]["path"]
classifier = pipeline("audio-classification", model=model_id)
classifier(audio_file)




# %%
#################### MANUAL METHOD ###############################
dataset = load_dataset("mozilla-foundation/common_voice_16_1", "ja", split="train")
sampling_rate = dataset.features["audio"].sampling_rate
model_id = "mageec/wave2vec2_capstone"
#%%
dataset = dataset.rename_column("locale", "label")
#%%

from transformers import AutoFeatureExtractor
import torch
#loading feature extractor to preprocess the audio file and return the input as PyTorch tensors
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
for i in range(5000, 5020):
    inputs = feature_extractor(dataset[i]["audio"]["array"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", max_length=int(15.0*sampling_rate), truncation=True)

    #passing inputs to the model and return the logits 
    from transformers import AutoModelForAudioClassification

    model = AutoModelForAudioClassification.from_pretrained(model_id)
    with torch.no_grad():
        logits = model(**inputs).logits

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
