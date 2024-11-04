import streamlit as st
import re
from gtts import gTTS
import os
import torch
import sounddevice as sd
import numpy as np
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Initialize the summarization pipeline
summ_obj = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

# Load pre-trained model and tokenizer for speech-to-text
model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Function to clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to summarize text
def summarize_text(text):
    cleaned_text = clean_text(text)
    summary = summ_obj(cleaned_text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# Function to generate audio from text using gTTS
def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    return "output.mp3"

# Function to transcribe audio using Wav2Vec2 model
def transcribe_audio(audio_input):
    input_values = tokenizer(audio_input, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription

# Function to process microphone input
def process_microphone_input(duration=10, sample_rate=16000):
    audio_frames = []
    try:
        audio_input = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        audio_input = np.squeeze(audio_input)
    except KeyboardInterrupt:
        pass
    return audio_input

def main():
    st.title("Handicap Assistant")

    # Text Summarization
    st.header("Text Summarization")
    text_input = st.text_area("Enter the text to summarize:")
    if st.button("Summarize"):
        if text_input:
            summary_text = summarize_text(text_input)
            st.subheader("Summary")
            st.write(summary_text)

    # Text-to-Speech (TTS)
    st.header("Text-to-Speech (TTS)")
    tts_text = st.text_input("Enter the text for TTS:")
    if st.button("Generate Audio"):
        if tts_text:
            audio_file = generate_audio(tts_text)
            st.subheader("TTS Audio")
            st.audio(audio_file, format='audio/mp3')

    # Speech-to-Text (STT)
    st.header("Speech-to-Text (STT)")
    st.info("Click the 'Record' button below to start recording.")
    if st.button("Record"):
        audio_data = process_microphone_input()
        transcription = transcribe_audio(audio_data)
        st.subheader("Transcription")
        st.write(transcription)

if __name__ == "__main__":
    main()
