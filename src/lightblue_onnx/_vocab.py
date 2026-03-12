vocab_phonemes = [
    'ˈ',
    'a','e','i','o','u',
    'b','v','d','h','z','χ','t','j','k','l','m','n','s','f','p','w','ʔ','ɡ','ʁ','ʃ','ʒ',
    ' ','.',',','!','?',"'",'"','-',':'
]

VOCAB_LIST = sorted(list(set(vocab_phonemes)))
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(VOCAB_LIST)}


def text_to_indices(text: str) -> list[int]:
    text = text.replace('g', 'ɡ').replace('r', 'ʁ')
    indices = []
    for c in text:
        if c not in CHAR_TO_ID:
            print(f"[Warning] Unknown char: '{c}' -> mapped to 0 (PAD)")
        indices.append(CHAR_TO_ID.get(c, 0))
    return indices
