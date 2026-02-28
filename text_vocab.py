import string

# Hardcoded phoneme vocabulary based on phonikud
vocab_phonemes = [
    'ˈ',
    'a','e','i','o','u',
    'b','v','d','h','z','χ','t','j','k','l','m','n','s','f','p','w','ʔ','ɡ','ʁ','ʃ','ʒ',
    ' ','.',',','!','?',"'",'"','-',':'
]

# Ensure unique and sorted
VOCAB_LIST = sorted(list(set(vocab_phonemes)))

# Mapping
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(VOCAB_LIST)}
ID_TO_CHAR = {i + 1: c for i, c in enumerate(VOCAB_LIST)}

def normalize_text(text):
    """
    Centralized normalization to ensure input text matches the vocab.
    Use this in Dataset AND Inference.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Fix g -> ɡ (ASCII to IPA)
    text = text.replace('g', 'ɡ') 
    # 2. Fix r -> ʁ (Common phonemizer mismatch)
    text = text.replace('r', 'ʁ') 
    
    return text

def text_to_indices(text):
    """Converts string to list of ints, handling normalization automatically."""
    text = normalize_text(text)
    indices = []
    for c in text:
        # Optional: Print warning if character is missing
        if c not in CHAR_TO_ID:
            print(f"[Warning] Unknown char found: '{c}' -> mapped to 0 (PAD)")
        indices.append(CHAR_TO_ID.get(c, 0)) # 0 is PAD / Unknown
    return indices