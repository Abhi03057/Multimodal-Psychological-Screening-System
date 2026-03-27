import requests
import numpy as np
from scipy.io import wavfile
import os

# Create a dummy audio file (3 seconds of a sine wave)
sample_rate = 22050
duration = 3.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

wav_path = "dummy_test.wav"
wavfile.write(wav_path, sample_rate, audio_data)

url = "http://127.0.0.1:8000/analysis/"

print(f"Testing {url} with a dummy {duration}s audio file...")

try:
    with open(wav_path, "rb") as f:
        files = {"audio_file": ("dummy_test.wav", f, "audio/wav")}
        data = {
            "phq9_score": 10,
            "gad7_score": 12
        }
        res = requests.post(url, files=files, data=data)

    print("Status Code:", res.status_code)
    try:
        print("Response:", res.json())
    except Exception as e:
        print("Raw text:", res.text)
finally:
    if os.path.exists(wav_path):
        os.remove(wav_path)
