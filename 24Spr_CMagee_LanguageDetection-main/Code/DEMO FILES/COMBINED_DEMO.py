#%%
import torch
import torch.nn.functional as F

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from transformers.pipelines.audio_utils import ffmpeg_read

import gradio as gr
import numpy as np
from transformers import pipeline



model_id = "mageec/whisper-05-01"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

sampling_rate = processor.feature_extractor.sampling_rate

bos_token_id = processor.tokenizer.all_special_ids[-106]
decoder_input_ids = torch.tensor([[1, bos_token_id]]).to(device)


def process_audio_file(file):
    with open(file, "rb") as f:
        inputs = f.read()

    audio = ffmpeg_read(inputs, sampling_rate)
    return audio


def transcribe(Microphone, File_Upload):
    warn_output = ""
    if (Microphone is not None) and (File_Upload is not None):
        warn_output = "WARNING: You've uploaded an audio file and used the microphone. " \
                      "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        file = Microphone

    elif (Microphone is None) and (File_Upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    elif Microphone is not None:
        file = Microphone
    else:
        file = File_Upload

    audio_data = process_audio_file(file)

    input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
    
    with torch.no_grad():
        logits = model.forward(input_features.to(device),decoder_input_ids=decoder_input_ids).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    probability = F.softmax(logits, dim=-1).max().item()

    lang_ids = processor.tokenizer.decode(pred_ids[0])
    
    lang_ids = lang_ids.replace("<|", " ").replace("|>", " ").split()[-1]
    language = LANGUAGES[lang_ids]

    return language, probability


transcriber = pipeline("automatic-speech-recognition", model="mageec/whisper-05-01")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]




demo1 = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="microphone", type='filepath'),
        gr.Audio(sources="upload", type='filepath'),
    ],
    outputs=[
        gr.Textbox(label="Language"),
        gr.Number(label="Probability"),
    ],
    title="Language Identification Model",
    description="Demo for Language Identification"
)



demo2 = gr.Interface(
    transcribe,
    gr.Audio(sources=["microphone"]),
    outputs = [gr.Textbox(label="Transcription")],
    title="Language Transcription Model",
    description="Demo for Transcription Model"
)



demo = gr.TabbedInterface([demo1, demo2], ["Language Identification", "Transcription"])
if __name__ == "__main__":
    demo.launch()