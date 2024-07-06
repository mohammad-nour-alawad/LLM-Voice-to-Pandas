import sounddevice as sd
import numpy as np
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration
import scipy.io.wavfile as wav
import time
import torch

# Constants
SAMPLING_RATE = 16000  # Sampling rate for recording
DURATION = 10  # Duration to record in seconds

# Load Whisper model
whisper_model = whisper.load_model("base").to('cuda' if torch.cuda.is_available() else 'cpu')

# Load fine-tuned BART model
bart_model_path = r"D:\#bd_ml\#project\Research internship 2\trained_model"  # replace with your actual model path
tokenizer = BartTokenizer.from_pretrained(bart_model_path)
model = BartForConditionalGeneration.from_pretrained(bart_model_path).to('cuda' if torch.cuda.is_available() else 'cpu')

def record_audio(duration, sampling_rate):
    print("Recording audio...")
    start_time = time.time()
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='int16')
    sd.wait()
    end_time = time.time()
    audio = audio.flatten()
    recording_time = end_time - start_time
    print(f"Recording Time: {recording_time:.2f} seconds")
    return audio, sampling_rate, recording_time

def save_audio(audio, sampling_rate, file_path):
    wav.write(file_path, sampling_rate, audio)

def transcribe_audio(audio_path):
    print("Transcribing audio...")
    start_time = time.time()
    result = whisper_model.transcribe(audio_path)
    end_time = time.time()
    transcription_time = end_time - start_time
    print(f"Transcription Time: {transcription_time:.2f} seconds")
    return result["text"], transcription_time

def translate_to_pandas(command_text):
    print("Translating text to Pandas command...")
    start_time = time.time()
    inputs = tokenizer(command_text, return_tensors="pt", max_length=512, truncation=True).to('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()
    translation_time = end_time - start_time
    print(f"Translation Time: {translation_time:.2f} seconds")
    return generated_text, translation_time

# Record audio
audio, sampling_rate, recording_time = record_audio(DURATION, SAMPLING_RATE)
audio_path = "recorded_command.wav"
save_audio(audio, sampling_rate, audio_path)

# Transcribe audio to text
transcribed_text, transcription_time = transcribe_audio(audio_path)
print("Transcribed Text:", transcribed_text)

# Translate text to Pandas command
pandas_command, translation_time = translate_to_pandas(transcribed_text)
print("Generated Pandas Command:", pandas_command)

# Summary of timings
processing_time = transcription_time + translation_time
total_time = recording_time + processing_time
print(f"\nSummary of Timings:")
print(f"Recording Time: {recording_time:.2f} seconds")
print(f"Transcription Time: {transcription_time:.2f} seconds")
print(f"Translation Time: {translation_time:.2f} seconds")
print(f"Processing Time (Transcription + Translation): {processing_time:.2f} seconds")
print(f"Total Time: {total_time:.2f} seconds")
