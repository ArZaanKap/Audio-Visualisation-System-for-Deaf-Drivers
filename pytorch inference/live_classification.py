from model_architecture import AudioCNN


import sounddevice as sd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import os

from collections import deque
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
model.load_state_dict(torch.load(model_path)) # map_location=device?####
model.eval()

# Compile model if using PyTorch 2.0+ (significant speedup)     -- triton??
if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cudnn.benchmark = True  # Find optimal algorithms
    
    #model = torch.compile(model, mode='reduce-overhead') # requires triton - huge speedup


class_names = ["horn", "other", "siren"] # list(test_dataset.classes)


# move transforms to gpu
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, 
    n_fft=1024, 
    hop_length=256, 
    n_mels=64
).to(device)

amplitude_to_db = T.AmplitudeToDB().to(device)

audio_buffer = deque(maxlen=CHUNK_SIZE*2) # O(1)

#preallocate tensors - avoid repeat allocation
waveform_buffer = torch.zeros(CHUNK_SIZE, device=device, dtype=torch.float32)

def callback(indata, frames, time_info, status):
    if status:
        print(status)

    # extract mono channel directly
    audio_buffer.extend(indata[:,0].astype(np.float32))


# List all available audio devices
#print("Available audio devices:")
#print(sd.query_devices())

# warm up - dummy inference to init CUDA kernels
print("Warming up...\n")
with torch.no_grad():
    dummy_input = torch.randn(1,1,64,432, device=device) ## correct?
    _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
print("warm up complete\n")

print("\nStarting Audio Stream...")

with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=2048):
    with torch.no_grad():
        while True:

            if len(audio_buffer) < CHUNK_SIZE:
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()

            # take required chunk - most recent data
            audio_segment = np.array(list(audio_buffer)[-CHUNK_SIZE:], dtype=np.float32)
            waveform_buffer.copy_(torch.from_numpy(audio_segment), non_blocking=True)

            # volume check on gpu - faster   # convert to tensor
            volume_window = waveform_buffer[-SHORT_WINDOW:]
            volume_sq = torch.mean(volume_window * volume_window)
            volume = torch.sqrt(volume_sq).item() #######?? synchronise

            if volume < VOLUME_THRESHOLD:
                print(f"Silence: {volume:.3f}")
                
                for _ in range(HOP_SIZE):
                    if audio_buffer:
                        audio_buffer.popleft()
                continue

            waveform_batch = waveform_buffer.unsqueeze(0)

            # create spectrogram
            mel_spec = mel_transform(waveform_batch)
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

            for _ in range(HOP_SIZE):
                if audio_buffer:
                    audio_buffer.popleft()



