import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

## Loading files
test_df = pd.read_csv("/Users/jackmcmorrow/Documents/capstone_backup/final_predictions/test_df.csv")
validation_df = pd.read_csv("/Users/jackmcmorrow/Documents/capstone_backup/final_predictions/validation_predictions.csv")
train_df = pd.read_csv("/Users/jackmcmorrow/Documents/capstone_backup/final_predictions/train_df.csv")
datasets = [test_df, validation_df, train_df]
dataset_names = ["Test Dataset", "Validation Dataset", "Train Dataset"]

for dataset in datasets:
    print(dataset.shape)
    print(dataset.info())

id2label = {
    0: "en",
    1: "ca",
    2: "rw",
    3: "be",
    4: "eo",
    5: "de",
    6: "fr",
    7: "kab",
    8: "es",
    9: "lg",
    10: "sw",
    11: "fa",
    12: "it",
    13: "mhr",
    14: "zh-CN",
    15: "ba",
    16: "ta",
    17: "ru",
    18: "eu",
    19: "th",
    20: "pt",
    21: "pl",
    22: "ja"
  }

def plot_confusion_matrix(true_labels, predicted_labels, save_path, name, label_mapping, add_labels=False):
    true_labels_mapped = [label_mapping[label] for label in true_labels]
    predicted_labels_mapped = [label_mapping[label] for label in predicted_labels]
    
    # Compute confusion matrix
    labels = list(label_mapping.values())
    cm = confusion_matrix(true_labels_mapped, predicted_labels_mapped)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion matrix: {name}')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if add_labels:
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
            
    plt.savefig(save_path)
    plt.show()  # Display the plot
    
# Getting metrics
def predict(df, name):
    print()
    print(name)
    print('--'*50)
    
    # Accuracy
    accuracy = accuracy_score(df["labels"], df["predictions"]) * 100
    print(f"Accuracy of {name}: {accuracy.__round__(2)}")
    print('--'*50)
    
    # F1 Score
    f1 = f1_score(df["labels"], df["predictions"], average='macro') * 100
    print(f"F1 Macros Score of {name}: {f1.__round__(2)}")
    print('--'*50)
    
    # Classification Report
    classification = classification_report(df["labels"], df["predictions"])
    print(f"Classification Report for {name}: \n{classification}")
    print('--'*50)

    # Confusion Matrix
    # labels = list(range(23))
    # plot_confusion_matrix(df["labels"], df["predictions"], labels)

labels = list(range(23))
for name, df in zip(dataset_names, datasets):
    predict(df, name)
    
    save_path = f"/Users/jackmcmorrow/Documents/GitHub/24Spr_CMagee_projectName/Code/confusion_matrix/{name}.png"
    save_path_labels = f"/Users/jackmcmorrow/Documents/GitHub/24Spr_CMagee_projectName/Code/confusion_matrix/{name}_labelled.png"
    plot_confusion_matrix(df["labels"], df["predictions"], save_path=save_path, name=name, label_mapping=id2label)
    plot_confusion_matrix(df["labels"], df["predictions"], save_path=save_path_labels, name=name, add_labels=True, label_mapping=id2label)