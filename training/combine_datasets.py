"""
combine_datasets.py — combine, clean, and phonemize TTS datasets.

Usage:
    python combine_datasets.py --config datasets.json
    python combine_datasets.py --config datasets.json --skip-phonemize
    python combine_datasets.py --config datasets.json --output out.csv --clean-output clean.csv

Config file format: see datasets.example.json
"""
import argparse
import json
import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_prefixed(x, prefix):
    x = str(x).strip()
    if os.path.isabs(x):
        return x
    if x.startswith(prefix):
        return x
    return os.path.join(prefix, x)


def _auto_text_col(df):
    for col in ("whisper_phonemes", "text", "phonemes"):
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_csv_dataset(spec):
    """
    Load a standard CSV dataset.

    Spec fields:
        csv              – path to CSV
        speaker_id       – integer (required unless splits is set)
        lang             – default "he"
        audio_dir        – prefix all filenames with this directory
        strip_prefix     – strip this literal string from filenames first
        filter_col       – boolean column; keep only True rows
        filename_col     – column with filenames (default "filename")
        text_col         – text/IPA column (auto-detected if omitted)
        csv_kwargs       – extra kwargs for pd.read_csv
        filename_template– e.g. "{index}.wav" to construct paths from another col
        extra_cols       – {col: value} added to every row
        splits           – [{pattern, speaker_id}]; partition by filename pattern
    """
    csv_path = spec["csv"]
    if not os.path.exists(csv_path):
        print(f"  Warning: {csv_path} not found, skipping.")
        return []

    df = pd.read_csv(csv_path, **spec.get("csv_kwargs", {}))
    print(f"  {os.path.basename(csv_path)}: {len(df)} rows")

    filter_col = spec.get("filter_col")
    if filter_col and filter_col in df.columns:
        df = df[df[filter_col].astype(bool)]
        print(f"    → after filter: {len(df)} rows")

    filename_col = spec.get("filename_col", "filename")

    strip = spec.get("strip_prefix")
    if strip and filename_col in df.columns:
        df[filename_col] = df[filename_col].str.replace(strip, "", regex=False)

    template = spec.get("filename_template")
    if template:
        df[filename_col] = df.apply(lambda row: template.format(**row.to_dict()), axis=1)

    audio_dir = spec.get("audio_dir")
    if audio_dir and filename_col in df.columns:
        df[filename_col] = df[filename_col].apply(lambda x: ensure_prefixed(x, audio_dir))

    if filename_col != "filename" and filename_col in df.columns:
        df = df.rename(columns={filename_col: "filename"})

    text_col = spec.get("text_col") or _auto_text_col(df)
    if text_col and text_col != "whisper_phonemes":
        df = df.rename(columns={text_col: "whisper_phonemes"})

    splits = spec.get("splits")
    if splits:
        result = []
        for s in splits:
            mask = df["filename"].str.contains(s["pattern"], regex=False)
            subset = df[mask].copy()
            if subset.empty:
                print(f"    Warning: no rows matched pattern '{s['pattern']}'")
                continue
            subset["speaker_id"] = s["speaker_id"]
            subset["lang"] = spec.get("lang", "he")
            print(f"    {s['pattern']}: {len(subset)} rows (speaker {s['speaker_id']})")
            result.append(subset)
        return result

    df["speaker_id"] = spec["speaker_id"]
    df["lang"] = spec.get("lang", "he")
    for col, val in spec.get("extra_cols", {}).items():
        df[col] = val
    return [df]


def load_libritts_dataset(spec):
    """
    Walk a LibriTTS-style directory tree (speaker/chapter/*.normalized.txt + *.wav).

    Spec fields:
        base_dir       – root of the LibriTTS download
        splits         – list of split names (e.g. ["train-clean-100"])
        speaker_offset – added to the numeric speaker directory name
        lang           – default "en"
    """
    rows = []
    base = spec["base_dir"]
    offset = spec.get("speaker_offset", 0)
    lang = spec.get("lang", "en")

    if not os.path.exists(base):
        print(f"  Warning: LibriTTS base_dir {base} not found, skipping.")
        return []

    for split in spec.get("splits", []):
        split_dir = os.path.join(base, split)
        if not os.path.exists(split_dir):
            print(f"  Warning: {split_dir} not found, skipping.")
            continue
        for speaker in os.listdir(split_dir):
            speaker_dir = os.path.join(split_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            try:
                spk_id = offset + int(speaker)
            except ValueError:
                continue
            for chapter in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter)
                if not os.path.isdir(chapter_dir):
                    continue
                for fname in os.listdir(chapter_dir):
                    if not fname.endswith(".normalized.txt"):
                        continue
                    stem = fname[: -len(".normalized.txt")]
                    wav = os.path.join(chapter_dir, stem + ".wav")
                    if not os.path.exists(wav):
                        continue
                    with open(os.path.join(chapter_dir, fname)) as fp:
                        text = fp.read().strip()
                    if text:
                        rows.append({"filename": wav, "whisper_phonemes": text,
                                     "speaker_id": spk_id, "wer_score": 0.0, "lang": lang})

    if not rows:
        print(f"  Warning: no LibriTTS data found in {base}")
        return []

    df = pd.DataFrame(rows)
    print(f"  LibriTTS: {len(df)} rows, {df['speaker_id'].nunique()} speakers")
    return [df]


_LOADERS = {
    "csv":      load_csv_dataset,
    "libritts": load_libritts_dataset,
}


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

COMMON_COLUMNS = ["filename", "whisper_phonemes", "speaker_id", "wer_score", "lang", "phonemized"]


def combine_csvs(config, output_path):
    dfs = []
    for spec in config.get("datasets", []):
        loader = _LOADERS.get(spec.get("type", "csv"))
        if loader is None:
            print(f"  Warning: unknown dataset type '{spec.get('type')}', skipping.")
            continue
        dfs.extend(loader(spec))

    if not dfs:
        print("No data loaded.")
        return None

    normalized = []
    for df in dfs:
        if "wer_score" not in df.columns:
            df["wer_score"] = 0.0
        if "lang" not in df.columns:
            df["lang"] = "he"
        cols = [c for c in COMMON_COLUMNS if c in df.columns]
        normalized.append(df[cols])

    combined = pd.concat(normalized, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"\nCombined → {output_path}  ({len(combined):,} rows)")
    return output_path


# ---------------------------------------------------------------------------
# Validation / cleaning
# ---------------------------------------------------------------------------

VALID_CHARS_HE = set("ˈaeiou" "bvdhztjklmnsfpwʔɡʁʃʒ" " .,!?'\"-:")
REPLACEMENTS_HE = {"g": "ɡ", "r": "ʁ"}


def _validate_hebrew(text):
    if not isinstance(text, str) or not text.strip():
        return "", False
    text = text.strip('"')
    for old, new in REPLACEMENTS_HE.items():
        text = text.replace(old, new)
    if set(text) - VALID_CHARS_HE:
        return text, False
    words = text.split()
    if len(words) >= 3 and any(words[i] == words[i+1] == words[i+2] for i in range(len(words)-2)):
        return text, False
    allowed_single = set("aeiou.,!?'\"-:")
    if any(len(w) == 1 and w not in allowed_single for w in words):
        return text, False
    return text, True


def _validate_raw(text):
    if not isinstance(text, str) or not text.strip():
        return "", False
    text = text.strip()
    if not any(c.isalpha() for c in text):
        return "", False
    words = text.split()
    if len(words) >= 3 and any(
        words[i].lower() == words[i+1].lower() == words[i+2].lower()
        for i in range(len(words)-2)
    ):
        return text, False
    return text, True


def clean_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    original = len(df)
    valid_mask, cleaned_texts, removed_by_lang = [], [], {}

    for _, row in df.iterrows():
        text = row["whisper_phonemes"]
        lang = row.get("lang", "he") if "lang" in df.columns else "he"
        cleaned, ok = _validate_hebrew(text) if lang == "he" else _validate_raw(text)
        valid_mask.append(ok)
        cleaned_texts.append(cleaned)
        if not ok:
            removed_by_lang[lang] = removed_by_lang.get(lang, 0) + 1

    df["whisper_phonemes"] = cleaned_texts
    df_clean = df[valid_mask].copy()
    removed = original - len(df_clean)

    print(f"\n--- Cleaning ---")
    print(f"  Original: {original:,}  Kept: {len(df_clean):,}  Removed: {removed:,} ({removed/original*100:.1f}%)")
    if removed_by_lang:
        print(f"  Removed by lang: {dict(sorted(removed_by_lang.items()))}")
    df_clean.to_csv(output_file, index=False)
    print(f"  Saved → {output_file}")


# ---------------------------------------------------------------------------
# Phonemization
# ---------------------------------------------------------------------------

def phonemize_dataset(input_file, espeak_lang_map=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    from data.text_vocab import normalize_text
    from tqdm import tqdm

    if espeak_lang_map is None:
        espeak_lang_map = {"en": "en-us", "es": "es", "de": "de", "fr": "fr"}

    df = pd.read_csv(input_file)
    if "phonemized" in df.columns and df["phonemized"].fillna(False).all():
        print("[Phonemize] All rows already phonemized.")
        return

    non_he = df["lang"] != "he"
    if "phonemized" in df.columns:
        non_he = non_he & ~df["phonemized"].fillna(False)
    if not non_he.any():
        print("[Phonemize] Nothing to phonemize.")
        return

    sep = Separator(phone="", word=" ", syllable="")
    for lang_code in df.loc[non_he, "lang"].unique():
        mask = non_he & (df["lang"] == lang_code)
        texts = [t.replace('"', "").replace("\u201c", "").replace("\u201d", "")
                 for t in df.loc[mask, "whisper_phonemes"]]
        espeak = espeak_lang_map.get(lang_code, lang_code)
        print(f"[Phonemize] {len(texts)} '{lang_code}' rows → espeak-ng ({espeak})")
        backend = EspeakBackend(espeak, preserve_punctuation=True,
                                with_stress=True, language_switch="remove-flags")
        ipa = []
        for i in tqdm(range(0, len(texts), 1000), desc=f"espeak ({lang_code})", unit="chunk"):
            ipa.extend(backend.phonemize(texts[i:i+1000], separator=sep, njobs=os.cpu_count()))
        ipa = [normalize_text(t, lang=lang_code) for t in ipa]
        df.loc[mask, "whisper_phonemes"] = ipa
        df.loc[mask, "phonemized"] = True

    df.loc[df["lang"] == "he", "phonemized"] = True
    non_he_all = df["lang"] != "he"
    df.loc[non_he_all, "whisper_phonemes"] = (
        df.loc[non_he_all, "whisper_phonemes"].str.replace(r'["\u201c\u201d]', "", regex=True)
    )
    df.to_csv(input_file, index=False)
    print(f"[Phonemize] Saved → {input_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="Path to JSON config file (see datasets.example.json)")
    parser.add_argument("--output", default=None,
                        help="Override combined CSV output path")
    parser.add_argument("--clean-output", default=None,
                        help="Override cleaned CSV output path")
    parser.add_argument("--skip-combine",    action="store_true")
    parser.add_argument("--skip-clean",      action="store_true")
    parser.add_argument("--skip-phonemize",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    output      = args.output      or config.get("output",       "combined_dataset.csv")
    clean_out   = args.clean_output or config.get("clean_output", "combined_dataset_cleaned.csv")

    combined_csv = output
    if not args.skip_combine:
        combined_csv = combine_csvs(config, output)

    if combined_csv and not args.skip_clean:
        clean_dataset(combined_csv, clean_out)

    if not args.skip_phonemize:
        phonemize_dataset(clean_out, espeak_lang_map=config.get("espeak_lang_map"))
