
import wave
import math
import struct
from pathlib import Path

# Create the assets folder if it doesn't exist
assets_folder = Path("assets")
assets_folder.mkdir(exist_ok=True)

file_path = assets_folder / "alert.wav"

sample_rate = 44100
duration = 2
frames = []

# Generate an alternating 880 Hz and 660 Hz alarm tone
for i in range(sample_rate * duration):
    frequency = 880 if (i // 11025) % 2 == 0 else 660
    sample = int(
        10000 * math.sin(2 * math.pi * frequency * i / sample_rate)
    )
    frames.append(struct.pack("<h", sample))

with wave.open(str(file_path), "wb") as audio:
    audio.setnchannels(1)
    audio.setsampwidth(2)
    audio.setframerate(sample_rate)
    audio.writeframes(b"".join(frames))

print(f"Alarm created successfully: {file_path.resolve()}")
