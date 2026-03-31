import torch
from torch.utils.data import Dataset
import soundfile as sf
import pandas as pd
import os
import random
import string
import glob
import numpy as np
import torchaudio
import torch.nn.functional as F
from data.text_vocab import text_to_indices, CHAR_TO_ID, ID_TO_CHAR, VOCAB_LIST
from data.audio_utils import ensure_sr

class TTSDataset(Dataset):
    def __init__(self, data_sources, sample_rate=44100, segment_size=None):
        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.data = [] # List of (wav_path, text)
        
        if isinstance(data_sources, str):
            data_sources = [data_sources]
            
        for source in data_sources:
            if os.path.isdir(source):
                # Case 1: Directory of audio files (recursive scan)
                print(f"Scanning for audio files in {source}...")
                audio_files = []
                valid_exts = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
                for root, _, files in os.walk(source):
                    for file in files:
                        if file.lower().endswith(valid_exts):
                            audio_files.append(os.path.join(root, file))
                
                print(f"Total audio files found in {source}: {len(audio_files)}")
                for w in audio_files:
                    self.data.append((w, "")) # Empty text
            elif os.path.isfile(source) and source.endswith(".csv"):
                # Case 2: Metadata CSV
                try:
                    # Assumes CSV with | separator: wav_path|text|normalized_text
                    df = pd.read_csv(source, sep='|', header=None, usecols=[0, 1], names=['wav_path', 'text'])
                    root_dir = os.path.dirname(source)
                    
                    # Check if "wavs" subdir exists, otherwise assume flat structure
                    wavs_dir = os.path.join(root_dir, 'wavs')
                    if not os.path.exists(wavs_dir):
                        wavs_dir = root_dir
                        
                    for _, row in df.iterrows():
                        wav_name = str(row['wav_path'])
                        text = str(row['text'])
                        
                        # Logic to guess extension if missing or check existence? 
                        # For now, we trust the metadata or check if file exists with common extensions
                        
                        if not os.path.splitext(wav_name)[1]:
                             # If no extension provided in CSV, try adding .wav, then others
                             base_path = os.path.join(wavs_dir, wav_name)
                             found_path = None
                             for ext in ['.wav', '.flac', '.mp3', '.ogg', '.m4a']:
                                 if os.path.exists(base_path + ext):
                                     found_path = base_path + ext
                                     break
                             if found_path:
                                 wav_path = found_path
                             else:
                                 # Default to .wav if not found
                                 wav_path = base_path + ".wav"
                        else:
                             wav_path = os.path.join(wavs_dir, wav_name)
                            
                        self.data.append((wav_path, text))
                        
                except Exception as e:
                    print(f"Error loading metadata {source}: {e}")
            else:
                print(f"Warning: Unknown data source or file not found: {source}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path, text = self.data[idx]
        
        try:
            wav_numpy, sr = sf.read(wav_path)           # float in [-1, 1]
        except Exception as e:
            print(f"Error loading audio file: {wav_path}")
            # Recursively try another sample
            return self.__getitem__((idx + 1) % len(self.data))

        wav = torch.from_numpy(wav_numpy).float()   # no extra scaling
        
        # Mono
        if wav.dim() > 1:
            wav = wav.mean(dim=1) # Average channels if stereo [T, C] or [T]
    
             
        # Resample
        if sr != self.sample_rate:
             wav = ensure_sr(wav, sr, self.sample_rate)
        
        if wav.dim() == 2: 
             wav = wav.mean(dim=0)
        else:
             wav = wav.squeeze(0) # ensure_sr returns [1, T] or [B, 1, T]
        
        wav_len = wav.shape[0]
        if self.segment_size is not None and wav_len > self.segment_size:
            max_start = wav_len - self.segment_size
            start = torch.randint(0, max_start + 1, (1,)).item()
            wav = wav[start:start + self.segment_size]

        # Text to IDs
        text_ids = torch.tensor(text_to_indices(text), dtype=torch.long)
        
        return wav, text_ids

def collate_fn(batch):
    # batch: list of (wav, text_ids)
    wavs = [item[0] for item in batch]
    
    # Pad Audio
    max_wav_len = max([w.shape[0] for w in wavs])
    wavs_padded = []
    
    for w in wavs:
        length = w.shape[0]
        pad_amt = max_wav_len - length
        # Pad at end
        wavs_padded.append(F.pad(w, (0, pad_amt)))
        
    wavs_padded = torch.stack(wavs_padded).unsqueeze(1) # [B, 1, T]
    
    return wavs_padded

