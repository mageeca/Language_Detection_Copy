import datasets
import ffmpeg
from datasets import load_dataset, Audio, load_from_disk

from transformers import pipeline

# Loading data
dataset = load_dataset("mozilla-foundation/common_voice_16_1", "ja", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]
print(audio_file)

model_name = "mageec/wave2vec2_capstone"
classifier = pipeline("audio-classification", model=model_name)
result = classifier(audio_file)
print(result)