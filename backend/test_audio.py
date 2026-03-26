import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from collections import Counter
import time
import sys
import os

from services.audio_model import analyze_audio, reset_buffer

# ══════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════
CHUNK_DURATION = 3      # seconds per recording chunk
SAMPLE_RATE    = 22050
OVERLAP_SLEEP  = 0.2    # short pause between chunks
CHUNK_FILE     = "chunk.wav"

# Emoji map for emotions
EMOTION_EMOJI = {
    "neutral"          : "😐",
    "calm"             : "😌",
    "happy"            : "😊",
    "sad"              : "😔",
    "angry"            : "😠",
    "fear"             : "😨",
    "disgust"          : "🤢",
    "surprise"         : "😲",
    "uncertain"        : "❓",
    "no speech detected": "🔇",
}

# Color codes (Windows CMD safe — uses simple separators instead of ANSI)
DIVIDER  = "─" * 55
DIVIDER2 = "═" * 55


# ══════════════════════════════════════════════════════
#  SESSION TRACKER
# ══════════════════════════════════════════════════════
session_emotions   = []   # all stable emotions this session
chunk_count        = 0


def record_chunk(filename=CHUNK_FILE, duration=CHUNK_DURATION, fs=SAMPLE_RATE):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    # scipy write expects int16
    write(filename, fs, (recording * 32767).astype(np.int16))


def confidence_bar(conf, width=20):
    """Visual confidence bar, e.g. ████████░░░░  0.72"""
    filled = int(conf * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {conf:.0%}"


def print_result(result, chunk_num):
    emotion   = result["emotion"]
    conf      = result["confidence"]
    energy    = result["speech_energy"]
    stability = result["stability"]
    flag      = result["behavior_flag"]
    trend     = result.get("trend", emotion)
    emoji     = EMOTION_EMOJI.get(emotion, "🔍")

    print(DIVIDER)
    print(f"  Chunk #{chunk_num:03d}   {emoji}  {emotion.upper()}")
    print(f"  Confidence : {confidence_bar(conf)}")
    print(f"  Energy     : {energy:<8}  Stability: {stability}")
    print(f"  Signal     : {flag}")
    if trend != emotion:
        print(f"  Raw→Stable : {trend}")
    print(DIVIDER)


def print_session_summary(emotions):
    if not emotions:
        return
    print(f"\n{DIVIDER2}")
    print("  SESSION SUMMARY")
    print(DIVIDER2)

    # Filter out non-speech
    speech = [e for e in emotions if e not in ("no speech detected", "uncertain")]

    if not speech:
        print("  No meaningful speech detected this session.")
    else:
        counts = Counter(speech)
        total  = len(speech)
        print(f"  Total chunks analysed : {len(emotions)}")
        print(f"  Speech chunks         : {total}")
        print()
        print("  Emotion breakdown:")
        for emotion, count in counts.most_common():
            pct   = count / total * 100
            emoji = EMOTION_EMOJI.get(emotion, "")
            bar   = "█" * int(pct / 5)
            print(f"    {emoji} {emotion:<12} {bar:<20} {pct:.1f}%")

        dominant = counts.most_common(1)[0][0]
        print()
        print(f"  Overall tendency  →  {EMOTION_EMOJI.get(dominant,'')} {dominant.upper()}")

    print(DIVIDER2)


# ══════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════
print(DIVIDER2)
print("  🎤  MULTIMODAL PSYCHOLOGICAL SCREENING SYSTEM")
print("  Audio Module — Real-Time Emotion Tracking")
print(DIVIDER2)
print("  Speak continuously.  Press Ctrl+C to stop.\n")

# Reset smoothing buffer at start of each session
reset_buffer()

try:
    while True:
        chunk_count += 1

        # ── Record ──────────────────────────────────
        print(f"  🔴 Recording chunk #{chunk_count:03d}...", end="\r")
        record_chunk()

        # ── Analyse ─────────────────────────────────
        result = analyze_audio(CHUNK_FILE)

        # ── Track session ───────────────────────────
        session_emotions.append(result["emotion"])

        # ── Display ─────────────────────────────────
        print_result(result, chunk_count)

        time.sleep(OVERLAP_SLEEP)

except KeyboardInterrupt:
    print("\n\n  🛑 Detection stopped.")
    print_session_summary(session_emotions)

except Exception as e:
    print(f"\n  ❌ Error: {e}")
    raise

finally:
    # Clean up temp file
    if os.path.exists(CHUNK_FILE):
        os.remove(CHUNK_FILE)