from model_architecture import AudioCNN


import sounddevice as sd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import os

import queue
import time


SAMPLE_RATE = 22050
CHUNK_DURATION = 5  # trained on 5s audio clips
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Use smaller hop for faster updates (process every 1 second instead of 5)
HOP_DURATION = 0.1  # seconds - how often to make a new prediction
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)

SHORT_WINDOW = int(0.1 * SAMPLE_RATE)

VOLUME_THRESHOLD = 0.03    # uses RMS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model path
model_dir = "pytorch training/models"
model_path = os.path.join(model_dir, "best_model_augmented_ONECYCLE_adam_drop2d2.pth")
print(model_path, "\n")

# architecture
model = AudioCNN()
model = model.to(device)
model.load_state_dict(torch.load(model_path)) # map_location=device?
model.eval()

# Compile model if using PyTorch 2.0+ (significant speedup)     -- triton??
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'


class_names = ["horn", "other", "siren"] # list(test_dataset.classes)


mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, 
    n_fft=1024, 
    hop_length=256, 
    n_mels=64
).to(device)

amplitude_to_db = T.AmplitudeToDB().to(device)

q = queue.Queue()
audio_buffer = np.array([], dtype=np.float32)


def callback(indata, frames, time_info, status):
    if status:
        print(status)

    # extract mono channel directly
    q.put(indata[:,0].astype(np.float32))


# List all available audio devices
#print("Available audio devices:")
#print(sd.query_devices())

print("Starting Audio Stream...")

with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=2048):
    with torch.no_grad():
        while True:

            try:
                audio_chunk = q.get(timeout=1.0)#??
            except queue.Empty:
                continue

            audio_buffer = np.concatenate([audio_buffer, audio_chunk])

            # keep buffer from growing too large
            if len(audio_buffer) > CHUNK_SIZE * 2:
                audio_buffer = audio_buffer[-(CHUNK_SIZE*2):]

            # process when enough audio is collected
            if len(audio_buffer) >= CHUNK_SIZE:

                # take required chunk - most recent data
                audio_segment = audio_buffer[-CHUNK_SIZE:]

                # volume check on gpu - faster   # convert to tensor
                waveform = torch.from_numpy(audio_segment).to(device, non_blocking=True)
                volume = torch.sqrt(torch.mean(waveform[-SHORT_WINDOW:]**2)).item()

                if volume < VOLUME_THRESHOLD:
                    print(f"Silence: {volume:.3f}")
                    audio_buffer = audio_buffer[HOP_SIZE:]
                    continue    # get next chunk

                start_time = time.perf_counter()
                audio_buffer = audio_buffer[HOP_SIZE:]    # shift buffer for next pred
    
                waveform = waveform.unsqueeze(0)    # add batch dim    

                # create spectrogram
                mel_spec = mel_transform(waveform)
                mel_db = amplitude_to_db(mel_spec)

                x = mel_db.unsqueeze(1)  # add channel dimension since model expects

                # forward
                logits = model(x)
                probs = torch.softmax(logits, dim=1)

                pred_idx = probs.argmax(dim=1).item()
                pred_class = class_names[pred_idx]
                confidence = 100*probs[0, pred_idx].item()

                inf_time = (time.perf_counter() - start_time) * 1000 # time in ms

                print(f"[{inf_time:.2f}ms]  {pred_class}: {confidence:.2f}%")





