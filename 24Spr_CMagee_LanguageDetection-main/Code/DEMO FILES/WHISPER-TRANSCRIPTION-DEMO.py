from transformers import pipeline
import gradio as gr
import gradio as gr
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="mageec/whisper-05-01")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]


demo = gr.Interface(
    transcribe,
    gr.Audio(sources=["microphone"]),
    outputs = [gr.Textbox(label="Transcription")],
    title="Language Transcription Model",
    description="Demo for Transcription Model"
)

demo.launch(share=True,server_port=7861)