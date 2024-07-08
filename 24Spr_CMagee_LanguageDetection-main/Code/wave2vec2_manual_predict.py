import datasets
from datasets import load_dataset, Audio, load_from_disk
import pandas as pd

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

# Loading Data
dataset = load_dataset("mozilla-foundation/common_voice_16_1", "en", split="train")
sampling_rate = dataset.features["audio"].sampling_rate
model_id = "mageec/wave2vec2_capstone"
dataset = dataset.rename_column("locale", "label")

# loading feature extractor to preprocess the audio file and return the input as PyTorch tensors
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

# Getting preprocess function
def preprocess_function(examples):
    
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    inputs = []
    
    for waveform in audio_arrays:
        input = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
             return_tensors="pt",
             padding=True
        )
        
        inputs.append(input)
        
    return inputs

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

model = AutoModelForAudioClassification.from_pretrained(model_id)
# inputs = preprocess_function(dataset)
# labels, probs = predict(dataset, model)
# print(labels, probs)

actual_labels = []
predicted_labels = []
predicted_probs = []

for i in range(5000, 5500):
    inputs = feature_extractor(dataset[i]["audio"]["array"], 
                               sampling_rate=feature_extractor.sampling_rate, 
                               return_tensors="pt",
                               max_length=int(15.0*sampling_rate), 
                               truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    #getting the label class with the highest probability, and using model's id2label mapping to convert into actual label 
    predicted_class_ids = torch.argmax(logits).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    predicted_label

    # Getting the probability scores
    softmax_scores = torch.softmax(logits, dim=1)
    predicted_prob_score = softmax_scores[0][predicted_class_ids].item()
    actual_label = dataset[i]['label']

    # Printing the predicted label along with its probability score
    print(f"Actual Label: {actual_label}, Predicted Label: {predicted_label}, Probability Score: {predicted_prob_score}")
    
    actual_labels.append(actual_label)
    predicted_labels.append(predicted_label)
    predicted_probs.append(predicted_prob_score)

print(actual_labels, predicted_labels, predicted_probs)

results_df = pd.DataFrame({
    "actual_label": actual_label,
    "prediction_label": predicted_labels,
    "predicted_probs": predicted_probs
})
print(results_df)
results_df.to_csv("Code/prediction_results/en_predictions.csv")