class Tokenizer:
    def __init__(self):
        # Default simple vocab (English + basic punctuation)
        # 0 is padding/unknown
        self.chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:;()\"'"
        self.char_to_id = {c: i+1 for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.chars) + 1

    def __call__(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join([self.id_to_char.get(i, "") for i in ids])

