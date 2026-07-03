import os
import glob
import pickle
from typing import List, Tuple, Dict
import numpy as np
from music21 import converter, instrument, note, chord

def parse_midi(filepath: str) -> List[Tuple[str, float]]:

    elements: List[Tuple[str, float]] = []
    try:
        score = converter.parse(filepath)
        parts = instrument.partitionByInstrument(score)

        part_to_use = parts.parts[0] if (parts and parts.parts) else score.flat

        for el in part_to_use.flat.notesAndRests:
            dur = float(el.duration.quarterLength)
            if isinstance(el, note.Note):
                elements.append((str(el.pitch), dur))
            elif isinstance(el, chord.Chord):
                token = ".".join(str(p) for p in el.pitches)
                elements.append((token, dur))

    except Exception as exc:
        print(f"  [WARN] Skipped '{os.path.basename(filepath)}': {exc}")

    return elements


def load_midi_dataset(midi_dir: str) -> List[Tuple[str, float]]:

    extensions = ["*.mid", "*.midi", "*.MID", "*.MIDI"]
    files: List[str] = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(midi_dir, "**", ext), recursive=True))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"\nNo MIDI files found in '{midi_dir}'.\n"
            "Place your .mid / .midi files there and try again."
        )

    print(f"\nFound {len(files)} MIDI file(s) in '{midi_dir}'")
    all_elements: List[Tuple[str, float]] = []

    for idx, filepath in enumerate(files, 1):
        print(f"  [{idx:>3}/{len(files)}] {os.path.basename(filepath)}")
        all_elements.extend(parse_midi(filepath))

    if not all_elements:
        raise ValueError("All MIDI files were empty or could not be parsed.")

    print(f"\nTotal elements extracted: {len(all_elements):,}\n")
    return all_elements

def build_vocabulary(
    elements: List[Tuple[str, float]],
) -> Tuple[Dict[str, int], Dict[int, str]]:

    unique_tokens = sorted(set(tok for tok, _ in elements))
    note_to_int   = {tok: idx for idx, tok in enumerate(unique_tokens)}
    int_to_note   = {idx: tok for tok, idx in note_to_int.items()}
    print(f"Vocabulary size: {len(note_to_int)} unique tokens")
    return note_to_int, int_to_note

def create_sequences(
    elements: List[Tuple[str, float]],
    note_to_int: Dict[str, int],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:

    tokens = [note_to_int[tok] for tok, _ in elements]

    X, y = [], []
    for i in range(len(tokens) - seq_len):
        X.append(tokens[i : i + seq_len])
        y.append(tokens[i + seq_len])

    X_arr = np.array(X, dtype=np.int32)
    y_arr = np.array(y, dtype=np.int32)

    print(f"Training sequences: {len(X_arr):,}  |  Input shape: {X_arr.shape}")
    return X_arr, y_arr

def save_data(data_dir: str, **kwargs) -> None:
    """Pickle each keyword argument to *data_dir/<name>.pkl*."""
    os.makedirs(data_dir, exist_ok=True)
    for name, obj in kwargs.items():
        path = os.path.join(data_dir, f"{name}.pkl")
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    saved = list(kwargs.keys())
    print(f"Saved {saved} → '{data_dir}'")


def load_data(data_dir: str, *names: str) -> dict:
    result: dict = {}
    for name in names:
        path = os.path.join(data_dir, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found.\n"
                "Run  python main.py preprocess  first."
            )
        with open(path, "rb") as fh:
            result[name] = pickle.load(fh)
    return result
