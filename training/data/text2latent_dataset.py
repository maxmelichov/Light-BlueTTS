import torch
from torch.utils.data import Dataset
import soundfile as sf
import pandas as pd
import os
import numpy as np
import torchaudio
import torch.nn.functional as F
import string
# Ensure these local modules exist in your project structure
from data.text_vocab import text_to_indices, VOCAB_LIST, normalize_text
from data.audio_utils import ensure_sr

class Text2LatentDataset(Dataset):
    """
    Dataset for text-to-latent training based on TTS/WildSpoof.
    
    Paper alignment:
    - Returns full utterances (cropping happens in training loop/collate).
    - Prioritizes Self-Reference (ref_wav = target_wav) to match the
      "randomly cropping input audio" strategy.
    """

    def __init__(
        self,
        metadata_path: str,
        sample_rate: int = 44100,
        hop_length: int = 512,
        max_wav_len: int = None,   # e.g., 44100 * 20
        max_text_len: int = None,
        cross_ref_prob: float = 0.5,  # Probability of cross-utterance same-speaker reference
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_wav_len = max_wav_len
        self.max_text_len = max_text_len
        self.cross_ref_prob = cross_ref_prob

        # --- 1. Load Metadata ---
        if not os.path.exists(metadata_path):
             raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Auto-detect separator
        self.df = pd.read_csv(metadata_path, sep=None, engine='python')

        # Initial filtering (loose)
        if 'wer_score' in self.df.columns:
            self.df = self.df[self.df['wer_score'] <= 0.0]
        
        # Normalize columns
        rename_map = {
            'filename': 'wav_path',
            'whisper_phonemes': 'text', 
            'file_id': 'wav_path', 
        }
        self.df.rename(columns=rename_map, inplace=True)

        # Fallback for headerless pipe-separated files
        if 'wav_path' not in self.df.columns or 'text' not in self.df.columns:
             try:
                 self.df = pd.read_csv(
                    metadata_path,
                    sep='|',
                    header=None, 
                    usecols=[0, 1],
                    names=['wav_path', 'text'], 
                    dtype={'wav_path': str, 'text': str}
                )
             except Exception:
                 pass

        self.root_dir = os.path.dirname(metadata_path)
        self.wavs_dir = self._detect_wav_dir(self.root_dir, self.df)

        # Drop empty text
        self.df = self.df.dropna(subset=['text'])

        # --- 2. Normalize Text & Validate ---
        self.df['text'] = self.df['text'].apply(normalize_text)
        
        # Check vocab coverage
        all_text = "".join(self.df["text"].astype(str).tolist())
        missing = set(all_text) - set(VOCAB_LIST)
        if missing:
            print(f"[Dataset] Warning: Missing chars in VOCAB: {repr(''.join(sorted(missing)))}")

        # Strict Filter: Drop rows producing unknown token ID (0)
        def has_unknown(text: str) -> bool:
            ids = text_to_indices(str(text))
            return any(i == 0 for i in ids)

        mask_valid_ids = ~self.df['text'].apply(has_unknown)
        if (~mask_valid_ids).sum() > 0:
            print(f"[Dataset] Dropping {(~mask_valid_ids).sum()} samples with unknown tokens.")
        self.df = self.df[mask_valid_ids].reset_index(drop=True)

        # --- 3. Length Filtering ---
        if self.max_text_len is not None:
            mask = self.df['text'].astype(str).str.len() <= self.max_text_len
            self.df = self.df[mask].reset_index(drop=True)

        if self.max_wav_len is not None:
            print(f"[Dataset] Filtering audio > {self.max_wav_len} samples...")
            mask_wav = self.df.apply(self._is_duration_ok, axis=1)
            self.df = self.df[mask_wav].reset_index(drop=True)

        # --- 4. Speaker ID & Strict WER Filter ---
        self._assign_speaker_ids()
        
        if 'wer_score' in self.df.columns:
            # Strict WER Filter: Only allow samples with wer_score <= 0.0 for ALL speakers
            len_before = len(self.df)
            self.df = self.df[self.df['wer_score'] <= 0.0].reset_index(drop=True)
            print(f"[Dataset] WER Filter dropped {len_before - len(self.df)} samples.")

        # --- 5. Indexing ---
        # Map speaker_id -> indices (useful for potential future cross-ref curriculum)
        self.speaker_to_indices = {k: v for k, v in self.df.groupby('speaker_id').indices.items()}
        print(f"[Dataset] Loaded {len(self.df)} samples across {len(self.speaker_to_indices)} speakers.")

    @property
    def speaker_ids(self):
        return self.df['speaker_id'].values

    def __len__(self):
        return len(self.df)

    def _detect_wav_dir(self, root, df):
        # Heuristic to find where wavs are stored
        if len(df) == 0: return root
        
        first_wav = str(df.iloc[0]['wav_path']).strip()
        if not first_wav.endswith('.wav'): first_wav += '.wav'
        
        # Check explicit 'wavs' subdir
        explicit_wavs = os.path.join(root, 'wavs')
        if os.path.exists(os.path.join(explicit_wavs, first_wav)):
            return explicit_wavs
            
        # Check root
        if os.path.exists(os.path.join(root, first_wav)):
            return root
            
        # Check first-level subdirectories
        if os.path.exists(root):
            for d in os.listdir(root):
                sub = os.path.join(root, d)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, first_wav)):
                    return sub
        return root

    def _resolve_path(self, wav_name):
        wav_name = str(wav_name).strip()
        if not wav_name.endswith('.wav') and not wav_name.endswith('.mp3'):
            wav_name += '.wav'
        if os.path.isabs(wav_name):
            return wav_name
        return os.path.join(self.wavs_dir, wav_name)

    def _is_duration_ok(self, row):
        try:
            wav_path = self._resolve_path(row['wav_path'])
            info = sf.info(wav_path)
            if info.samplerate == 0: return False
            est_samples = (info.frames / info.samplerate) * self.sample_rate
            return est_samples <= self.max_wav_len
        except:
            return True

    def _assign_speaker_ids(self):
        self.df['speaker_id'] = self.df['speaker_id'].astype(int)

    def _load_wav_by_index(self, idx):
        row = self.df.iloc[idx]
        wav_path = self._resolve_path(row['wav_path'])

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"File not found: {wav_path}")

        wav_numpy, sr = sf.read(wav_path, dtype='float32')
        wav = torch.from_numpy(wav_numpy).float()
        
        # Stereo to Mono
        if wav.dim() > 1: 
            wav = wav.mean(dim=1)
        
        # Resample
        wav = ensure_sr(wav, sr, self.sample_rate, device='cpu')
        
        # Ensure [T] shape
        if wav.dim() != 1:
            if wav.dim() == 2: wav = wav.mean(dim=0)
            else: raise RuntimeError(f"Bad shape {wav.shape} for {wav_path}")

        # Safety: NaN/Inf
        if not torch.isfinite(wav).all():
            wav = torch.nan_to_num(wav)

        # Pad to at least one hop_length (prevents issues in latent encoding)
        if wav.numel() < self.hop_length:
            wav = F.pad(wav, (0, self.hop_length - wav.numel()))
        else:
            # Trim to nearest hop
            valid_len = (wav.numel() // self.hop_length) * self.hop_length
            wav = wav[:valid_len]
            
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text']).strip()
        speaker_id = int(row['speaker_id'])
        
        # 1. Load Target Wav
        wav = self._load_wav_by_index(idx)
        
        # 2. Reference Audio Strategy (Zero-Shot Training)
        # SupertonicTTS (Sec 4.2): reference from same speaker.
        # Self-ref: random crop of target (masking prevents copying).
        # Cross-ref: different utterance from same speaker (teaches
        #   the reference encoder to extract general speaker identity).
        # Mix of both gives alignment stability (self-ref) + zero-shot
        # generalization (cross-ref).
        
        same_speaker_indices = self.speaker_to_indices.get(speaker_id, [])
        # Cross-utterance reference: pick a different utterance from same speaker
        use_cross_ref = (
            np.random.random() < self.cross_ref_prob
            and len(same_speaker_indices) > 1
        )
        
        if use_cross_ref:
            # Pick a random utterance from the same speaker, excluding self
            candidates = [i for i in same_speaker_indices if i != idx]
            if candidates:
                ref_idx = np.random.choice(candidates)
                try:
                    ref_wav = self._load_wav_by_index(ref_idx)
                    is_self_ref = False
                except Exception:
                    # Fallback to self-reference on load failure
                    ref_wav = wav.clone()
                    is_self_ref = True
            else:
                # Only one utterance for this speaker, fallback to self-ref
                ref_wav = wav.clone()
                is_self_ref = True
        else:
            # Self-reference: clone target (cropping happens in training loop)
            ref_wav = wav.clone()
            is_self_ref = True
        
        ref_speaker_id = speaker_id

        # 3. Text to Indices
        text_ids = torch.tensor(text_to_indices(text), dtype=torch.long)
        
        return wav, text_ids, speaker_id, ref_wav, is_self_ref, ref_speaker_id


def collate_text2latent(batch):
    """
    Collate function to pad batches.
    Returns:
       wavs_padded:     [B, 1, T]
       texts_padded:    [B, T_txt]
       text_masks:      [B, 1, T_txt]
       wav_lengths:     [B]
       speaker_ids:     [B]
       ref_wavs_padded: [B, 1, T_ref]
       ref_lengths:     [B]
       is_self_ref:     [B] (bool)
       ref_speaker_ids: [B]
    """
    # Unpack
    wavs, texts, spk_ids, ref_wavs, is_self, ref_spk_ids = zip(*batch)

    # 1. Pad Target Wavs
    max_wav = max(w.shape[0] for w in wavs)
    wavs_padded = []
    wav_lens = []
    for w in wavs:
        pad = max_wav - w.shape[0]
        wavs_padded.append(F.pad(w, (0, pad)))
        wav_lens.append(w.shape[0])
    
    wavs_padded = torch.stack(wavs_padded).unsqueeze(1) # [B, 1, T]
    wav_lens = torch.tensor(wav_lens, dtype=torch.long)

    # 2. Pad Reference Wavs
    max_ref = max(w.shape[0] for w in ref_wavs)
    ref_padded = []
    ref_lens = []
    for w in ref_wavs:
        pad = max_ref - w.shape[0]
        ref_padded.append(F.pad(w, (0, pad)))
        ref_lens.append(w.shape[0])
        
    ref_padded = torch.stack(ref_padded).unsqueeze(1) # [B, 1, T_ref]
    ref_lens = torch.tensor(ref_lens, dtype=torch.long)
    
    # 3. Pad Text
    max_txt = max(t.shape[0] for t in texts)
    txt_padded = []
    txt_masks = []
    for t in texts:
        pad = max_txt - t.shape[0]
        txt_padded.append(F.pad(t, (0, pad), value=0)) # 0 is usually padding ID
        
        mask = torch.zeros(max_txt, dtype=torch.float32)
        mask[:t.shape[0]] = 1.0
        txt_masks.append(mask)
        
    txt_padded = torch.stack(txt_padded)          # [B, T_txt]
    txt_masks = torch.stack(txt_masks).unsqueeze(1) # [B, 1, T_txt]
    
    # 4. Metadata
    spk_ids = torch.tensor(spk_ids, dtype=torch.long)
    is_self = torch.tensor(is_self, dtype=torch.bool)
    ref_spk_ids = torch.tensor(ref_spk_ids, dtype=torch.long)

    return wavs_padded, txt_padded, txt_masks, wav_lens, spk_ids, ref_padded, ref_lens, is_self, ref_spk_ids