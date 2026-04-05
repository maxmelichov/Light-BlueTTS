"""
Multilingual IPA vocabulary for TTS, built on the piper phoneme set.

Layout:
  0..156   = Piper core symbols (exact IDs from rhasspy/piper-checkpoints)
  157..243 = Extended symbols (uppercase letters, additional IPA, punctuation)
  244..383 = Language tokens (140 slots)

VOCAB_SIZE = 384  (fixed, embedding-friendly: 384 = 6 × 64)

Special indices:
  PAD_ID = 0  (_)   padding / unknown
  BOS_ID = 1  (^)   begin-of-sequence (piper convention)
  EOS_ID = 2  ($)   end-of-sequence   (piper convention)

Adding a new language:
  1. Add one line to LANG_ID below
  2. Add training data with a 'lang' column in the CSV
  3. Resume training — no architecture change needed
"""

import re
from unicodedata import normalize as uni_normalize

# ============================================================
# Piper phoneme map (exact IDs from rhasspy/piper-checkpoints)
# Source: en/en_US/ljspeech/medium/config.json  phoneme_id_map
# ============================================================
_PIPER_MAP: dict[str, int] = {
    "_": 0,   # PAD
    "^": 1,   # BOS
    "$": 2,   # EOS
    " ": 3,
    "!": 4, "'": 5, "(": 6, ")": 7, ",": 8, "-": 9, ".": 10,
    ":": 11, ";": 12, "?": 13,
    "a": 14, "b": 15, "c": 16, "d": 17, "e": 18, "f": 19,
    "h": 20, "i": 21, "j": 22, "k": 23, "l": 24, "m": 25, "n": 26,
    "o": 27, "p": 28, "q": 29, "r": 30, "s": 31, "t": 32, "u": 33,
    "v": 34, "w": 35, "x": 36, "y": 37, "z": 38,
    "æ": 39, "ç": 40, "ð": 41, "ø": 42, "ħ": 43, "ŋ": 44, "œ": 45,
    "ǀ": 46, "ǁ": 47, "ǂ": 48, "ǃ": 49,
    "ɐ": 50, "ɑ": 51, "ɒ": 52, "ɓ": 53, "ɔ": 54, "ɕ": 55,
    "ɖ": 56, "ɗ": 57, "ɘ": 58, "ə": 59, "ɚ": 60, "ɛ": 61,
    "ɜ": 62, "ɞ": 63, "ɟ": 64, "ɠ": 65, "ɡ": 66, "ɢ": 67,
    "ɣ": 68, "ɤ": 69, "ɥ": 70, "ɦ": 71, "ɧ": 72, "ɨ": 73,
    "ɪ": 74, "ɫ": 75, "ɬ": 76, "ɭ": 77, "ɮ": 78, "ɯ": 79,
    "ɰ": 80, "ɱ": 81, "ɲ": 82, "ɳ": 83, "ɴ": 84, "ɵ": 85,
    "ɶ": 86, "ɸ": 87, "ɹ": 88, "ɺ": 89, "ɻ": 90, "ɽ": 91,
    "ɾ": 92, "ʀ": 93, "ʁ": 94, "ʂ": 95, "ʃ": 96, "ʄ": 97,
    "ʈ": 98, "ʉ": 99, "ʊ": 100, "ʋ": 101, "ʌ": 102, "ʍ": 103,
    "ʎ": 104, "ʏ": 105, "ʐ": 106, "ʑ": 107, "ʒ": 108, "ʔ": 109,
    "ʕ": 110, "ʘ": 111, "ʙ": 112, "ʛ": 113, "ʜ": 114, "ʝ": 115,
    "ʟ": 116, "ʡ": 117, "ʢ": 118, "ʲ": 119,
    "ˈ": 120, "ˌ": 121, "ː": 122, "ˑ": 123, "˞": 124,
    "β": 125, "θ": 126, "χ": 127, "ᵻ": 128, "ⱱ": 129,
    "0": 130, "1": 131, "2": 132, "3": 133, "4": 134,
    "5": 135, "6": 136, "7": 137, "8": 138, "9": 139,
    "\u0327": 140,  # ̧  combining cedilla
    "\u0303": 141,  # ̃  combining tilde
    "\u032A": 142,  # ̪  combining dental below
    "\u032F": 143,  # ̯  combining inverted breve below
    "\u0329": 144,  # ̩  combining vertical line below
    "ʰ": 145, "ˤ": 146, "ε": 147, "↓": 148, "#": 149,
    '"': 150, "↑": 151,
    "\u033A": 152,  # ̺  combining inverted bridge below
    "\u033B": 153,  # ̻  combining square below
    "g": 154, "ʦ": 155, "X": 156,
}

# ============================================================
# Extended symbol map (indices 157..243)
# Covers uppercase letters, additional IPA symbols, and punctuation
# from: _letters, _letters_ipa, _punctuation
# ============================================================
_EXTENDED_MAP: dict[str, int] = {
    # --- Uppercase letters (A-Z, X already at 156) ---
    "A": 157, "B": 158, "C": 159, "D": 160, "E": 161, "F": 162, "G": 163,
    "H": 164, "I": 165, "J": 166, "K": 167, "L": 168, "M": 169, "N": 170,
    "O": 171, "P": 172, "Q": 173, "R": 174, "S": 175, "T": 176, "U": 177,
    "V": 178, "W": 179, "Y": 180, "Z": 181,
    # --- Additional IPA symbols ---
    "ʤ": 182,  # voiced postalveolar affricate (dʒ digraph)
    "ɝ": 183,  # open-mid central rhotic vowel
    "ʧ": 184,  # voiceless postalveolar affricate (tʃ digraph)
    "ʼ": 185,  # ejective marker (modifier letter apostrophe)
    "ʴ": 186,  # modifier letter small turned r
    "ʱ": 187,  # modifier letter small h with hook
    "ʷ": 188,  # modifier letter small w (labialization diacritic)
    "ˠ": 189,  # modifier letter small gamma (velarization diacritic)
    "→": 190,  # rightwards arrow (level tone)
    "↗": 191,  # north east arrow (rising tone)
    "↘": 192,  # south east arrow (falling tone)
    # --- Additional punctuation ---
    "¡": 193,  # inverted exclamation mark
    "¿": 194,  # inverted question mark
    "…": 195,  # horizontal ellipsis
    "«": 196,  # left-pointing double angle quotation mark
    "»": 197,  # right-pointing double angle quotation mark
    "*": 198,  # asterisk
    "~": 199,  # tilde
    "/": 200,  # solidus
    "\\": 201, # reverse solidus
    "&": 202,  # ampersand
    # --- Combining IPA diacritics (203..223) ---
    "\u0361": 203,  # ͡  combining double inverted breve (affricate tie, e.g. t͡ʃ)
    "\u035C": 204,  # ͜  combining double breve below (alternative tie bar)
    "\u0325": 205,  # ̥  combining ring below (devoiced)
    "\u032C": 206,  # ̬  combining caron below (voiced)
    "\u0339": 207,  # ̹  combining right half ring below (more rounded)
    "\u031C": 208,  # ̜  combining left half ring below (less rounded)
    "\u031D": 209,  # ̝  combining up tack below (raised)
    "\u031E": 210,  # ̞  combining down tack below (lowered)
    "\u031F": 211,  # ̟  combining plus sign below (advanced)
    "\u0320": 212,  # ̠  combining minus sign below (retracted)
    "\u0330": 213,  # ̰  combining tilde below (creaky voice)
    "\u0334": 214,  # ̴  combining tilde overlay (nasalization)
    "\u031A": 215,  # ̚  combining left angle above (no audible release)
    "\u0318": 216,  # ̘  combining left tack below (advanced tongue root)
    "\u0319": 217,  # ̙  combining right tack below (retracted tongue root)
    "\u0348": 218,  # ͈  combining double vertical line below (fortis)
    "\u0306": 219,  # ̆  combining breve (extra short)
    "\u0308": 220,  # ̈  combining diaeresis above (centralized)
    "\u031B": 221,  # ̛  combining horn
    "\u0324": 222,  # ̤  combining diaeresis below (breathy voice)
    "\u033C": 223,  # ̼  combining seagull below (linguolabial)
    # --- Modifier letters (224..232) ---
    "\u02C0": 224,  # ˀ  modifier letter glottal stop
    "\u02C1": 225,  # ˁ  modifier letter reversed glottal stop
    "\u02BE": 226,  # ʾ  modifier letter right half ring
    "\u02BF": 227,  # ʿ  modifier letter left half ring
    "\u02BB": 228,  # ʻ  modifier letter turned comma
    "\u02C9": 229,  # ˉ  modifier letter macron (level tone)
    "\u02CA": 230,  # ˊ  modifier letter acute accent (rising tone)
    "\u02CB": 231,  # ˋ  modifier letter grave accent (falling tone)
    "\u02C6": 232,  # ˆ  modifier letter circumflex accent
    # --- Tone bar letters (233..237) ---
    "\u02E5": 233,  # ˥  extra-high tone bar
    "\u02E6": 234,  # ˦  high tone bar
    "\u02E7": 235,  # ˧  mid tone bar
    "\u02E8": 236,  # ˨  low tone bar
    "\u02E9": 237,  # ˩  extra-low tone bar
    # --- Additional combining accent marks (238..243) ---
    "\u0300": 238,  # ̀  combining grave accent
    "\u0301": 239,  # ́  combining acute accent
    "\u0302": 240,  # ̂  combining circumflex accent
    "\u0304": 241,  # ̄  combining macron (mid tone)
    "\u030C": 242,  # ̌  combining caron
    "\u0307": 243,  # ̇  combining dot above
}

# ============================================================
# Region constants
# ============================================================
PIPER_REGION_END  = 156   # last piper index
LANG_REGION_START = 244   # first language token index
LANG_REGION_SIZE  = 140   # slots for 140 languages
VOCAB_SIZE        = LANG_REGION_START + LANG_REGION_SIZE  # 384

# Special token IDs
PAD_ID  = 0   # "_"
BOS_ID  = 1   # "^"
EOS_ID  = 2   # "$"

# ============================================================
# Language tokens  (indices 244..383)
# To add a language: add one line here, nothing else changes.
# ============================================================
LANG_ID: dict[str, int] = {
    "he": LANG_REGION_START + 0,   # 244 — Hebrew
    "en": LANG_REGION_START + 1,   # 245 — English
    # --- add new languages below, in any order ---
    "es": LANG_REGION_START + 2,   # 246 — Spanish
    # "fr": LANG_REGION_START + 3,  # 247 — French
    # "ar": LANG_REGION_START + 4,  # 248 — Arabic
    # "pt": LANG_REGION_START + 5,  # 249 — Portuguese
    # "ko": LANG_REGION_START + 6,  # 250 — Korean
    # "ja": LANG_REGION_START + 7,  # 251 — Japanese
    "de": LANG_REGION_START + 8,   # 252 — German
    "it": LANG_REGION_START + 9,   # 253 — Italian
    # "ru": LANG_REGION_START + 10, # 254 — Russian
    # "zh": LANG_REGION_START + 11, # 255 — Chinese
    # "hi": LANG_REGION_START + 12, # 256 — Hindi
    # "tr": LANG_REGION_START + 13, # 257 — Turkish
    # "xx": LANG_REGION_START + 139,# 383 — last slot
}

LANG_NAMES: dict[int, str] = {v: k for k, v in LANG_ID.items()}

# ============================================================
# Lookup tables
# ============================================================
CHAR_TO_ID: dict[str, int] = {**_PIPER_MAP, **_EXTENDED_MAP}

ID_TO_CHAR: dict[int, str] = {v: k for k, v in CHAR_TO_ID.items()}
for _lang_name, _lang_idx in LANG_ID.items():
    ID_TO_CHAR[_lang_idx] = f"<{_lang_name}>"

# All characters that have a valid mapping (used for coverage reporting in dataset)
VOCAB_LIST: list[str] = list(CHAR_TO_ID.keys())

# ============================================================
# Safety assertions
# ============================================================
assert VOCAB_SIZE == 384
assert len(_PIPER_MAP) == 157, f"Piper map must have 157 entries, got {len(_PIPER_MAP)}"
assert max(_PIPER_MAP.values()) == PIPER_REGION_END, \
    f"Piper max ID should be {PIPER_REGION_END}"
assert all(v < LANG_REGION_START for v in _PIPER_MAP.values()), \
    "Piper IDs must not overlap language region"
assert all(PIPER_REGION_END < v < LANG_REGION_START for v in _EXTENDED_MAP.values()), \
    "Extended IDs must be in range 157..243"
assert len(set(_EXTENDED_MAP.values())) == len(_EXTENDED_MAP), \
    "Extended map has duplicate IDs"
assert not (set(_PIPER_MAP.values()) & set(_EXTENDED_MAP.values())), \
    "Piper and extended maps share IDs"


# ============================================================
# Public API
# ============================================================

def text_to_indices(text: str, lang: str = "he") -> list[int]:
    """
    Convert an IPA phoneme string to vocab indices with a language token prepended.

    Args:
        text: IPA string (output of a G2P / phonemizer tool)
        lang: language code ("he", "en", "es", ...)
    Returns:
        list of ints — first element is the language token (244+),
        followed by per-character IDs. Unknown chars map to PAD_ID (0).
    Raises:
        ValueError: if lang is not registered in LANG_ID
    """
    if lang not in LANG_ID:
        raise ValueError(
            f"Unknown language '{lang}'. Available: {list(LANG_ID.keys())}. "
            f"Add it to LANG_ID in text_vocab.py"
        )
    lang_token = LANG_ID[lang]
    return [lang_token] + [CHAR_TO_ID.get(ch, PAD_ID) for ch in text]


def indices_to_text(indices: list[int]) -> str:
    """Convert indices back to a readable string (for debugging)."""
    return "".join(ID_TO_CHAR.get(i, "?") for i in indices)


def normalize_text(text: str, lang: str = "he") -> str:
    """
    Normalize text before tokenization.
    Applies Unicode normalization and common phoneme substitutions.

    Args:
        text: IPA or raw-text string to normalize.
        lang: language code — Hebrew-specific substitutions (r→ʁ, g→ɡ)
              are only applied when lang == "he".
    """
    text = text.strip()
    text = uni_normalize("NFD", text)

    # Universal replacements (fancy quotes/dashes → ASCII equivalents)
    replacements = {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "´": "'", "`": "'",
        "–": "-", "‑": "-", "—": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Hebrew phonemizer output normalization
    # (phonikud sometimes emits ASCII r/g instead of IPA ʁ/ɡ)
    if lang == "he":
        text = text.replace("r", "ʁ")
        text = text.replace("g", "ɡ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_coverage(text: str) -> dict:
    """Debug: report which chars in text are missing from vocab."""
    known, unknown = [], []
    for ch in text:
        (known if ch in CHAR_TO_ID else unknown).append(ch)
    return {
        "total": len(text),
        "known": len(known),
        "unknown": len(unknown),
        "missing_chars": sorted(set(unknown)),
        "missing_codepoints": [f"U+{ord(c):04X}" for c in sorted(set(unknown))],
    }


def get_supported_languages() -> list[str]:
    return list(LANG_ID.keys())


# ============================================================
# Import diagnostics
# ============================================================
print(
    f"[Vocab] VOCAB_SIZE={VOCAB_SIZE} | "
    f"piper={len(_PIPER_MAP)} (0..{PIPER_REGION_END}) | "
    f"extended={len(_EXTENDED_MAP)} (157..{max(_EXTENDED_MAP.values())}) | "
    f"lang_slots={LANG_REGION_SIZE} ({LANG_REGION_START}..{VOCAB_SIZE - 1}) | "
    f"active_langs={list(LANG_ID.keys())}"
)
