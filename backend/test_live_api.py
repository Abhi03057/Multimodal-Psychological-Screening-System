import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import requests
import time
import os

# Configuration
CHUNK_DURATION = 3      
SAMPLE_RATE    = 22050
CHUNK_FILE     = "live_chunk.wav"
API_URL        = "http://127.0.0.1:8000/analysis/"

# ══════════════════════════════════════════════════════
#  MIC SELECTION (hardcoded to Realtek Microphone Array)
# ══════════════════════════════════════════════════════
device_id = 3  # Microphone Array (Realtek)

def record_chunk(filename=CHUNK_FILE, duration=CHUNK_DURATION, fs=SAMPLE_RATE, dev=None):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32", device=dev)
    sd.wait()
    
    # Debug: Check if the mic actually heard anything
    peak_amplitude = np.max(np.abs(recording))
    print(f"  🛠️ [Debug] Mic Peak Amplitude: {peak_amplitude:.5f}")
    if peak_amplitude < 0.001:
        print("  ⚠️ WARNING: Still picking up near silence! Make sure Windows Privacy Settings allow microphone access for Python apps.")
        print("  (Settings > Privacy & security > Microphone > 'Let desktop apps access your microphone' = ON)")
    
    # Save as 16-bit PCM WAV
    write(filename, fs, (recording * 32767).astype(np.int16))

print("\n" + "=" * 60)
print("🎤 LIVE API TESTING SCRIPT STARTED")
print("=" * 60)
print("Press Ctrl+C to stop.\n")

try:
    chunk_count = 0
    while True:
        chunk_count += 1
        print(f"\n🔴 [Chunk #{chunk_count}] Recording for {CHUNK_DURATION} seconds...")
        record_chunk(dev=device_id)
        
        print("  ✅ Recorded. Sending to API...")
        data = {"phq9_score": 8, "gad7_score": 6}
        
        try:
            with open(CHUNK_FILE, "rb") as f:
                res = requests.post(API_URL, files={"audio_file": (CHUNK_FILE, f, "audio/wav")}, data=data)
                
            if res.status_code == 200:
                result = res.json()
                emotion = result.get('detected_emotion', 'unknown')
                energy = result['audio_features'].get('energy', 'unknown')
                
                print(f"  👉 Detected Emotion: {emotion.upper()}")
                print(f"  ⚡ Vocal Energy: {energy}")
                
                for line in result['report'].split('\n'):
                    if any(word in line for word in ["Interpretation:", "Recommendation:", "The ", "Speech ", "If ", "This "]):
                        if line.strip():
                            print(f"      {line.strip()}")
            else:
                print(f"  ❌ API Error: {res.status_code} - {res.text}")
                
        except requests.exceptions.ConnectionError:
            print("  ❌ Could not connect to the server. Is it running?")
            
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 Live testing stopped.")
finally:
    if os.path.exists(CHUNK_FILE):
        os.remove(CHUNK_FILE)
