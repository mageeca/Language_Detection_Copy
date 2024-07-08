#%%
import numpy as np
import librosa 
from datasets import load_dataset, Audio
import matplotlib.pyplot as plt
#%%
data = load_dataset("mageec/test_data_subset", split="train", streaming = True)
data = data.shuffle(seed=102)
sample = next(iter(data))

print(sample)
sentence = sample["sentence"]
#print(sentence)
#%%
sample_audio = sample["audio"]
print(sample_audio)
#%%
array = sample_audio["array"]
D = librosa.stft(array)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#%%
plt.figure(figsize=(12, 4))
plt.style.use('seaborn-v0_8-whitegrid') 
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar()

plt.title('Spectogram of Audio Recording in English', fontsize = 18)
plt.figtext(0.05, 0.0000000000000001, 'SENTENCE:"He co-discovered a couple of comets."', ha='left', fontsize=12)

plt.show()

#%%
import matplotlib.pyplot as plt
S = librosa.feature.melspectrogram(y=array, sr=48000, n_mels=128)
log_S = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize=(12,4))
librosa.display.specshow(log_S, sr=48000, x_axis='time', y_axis='mel')
plt.title('Mel Power Spectrogram of Audio Recording in English', fontsize = 18)
plt.figtext(0.05, 0.0000000000001, 'SENTENCE: "He co-discovered a couple of comets."', ha='left', fontsize=12)
plt.style.use('seaborn-v0_8-whitegrid')
# %%
######WAVEFORM########
plt.figure(figsize=(12, 4))
plt.style.use('seaborn-v0_8-whitegrid') 
librosa.display.waveshow(array, sr=sample_audio["sampling_rate"],color="purple")
plt.title('Waveform of Audio Recording in English', fontsize = 18)
plt.figtext(0.05, 0.0000000000001, 'SENTENCE: "He co-discovered a couple of comets."', ha='left', fontsize=12)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# %%

# %%
