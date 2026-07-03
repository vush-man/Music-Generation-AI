import os
from typing import List, Optional
import numpy as np
import tensorflow as tf
from music21 import chord, instrument, note, stream, tempo
import config
from data_preprocessing import create_sequences, load_data

def _sample(probs: np.ndarray, temperature: float = 1.0) -> int:

    probs = probs.astype("float64")
    log_probs = np.log(probs + 1e-10) / temperature
    exp_probs = np.exp(log_probs - log_probs.max())   # numeric stability
    exp_probs /= exp_probs.sum()
    return int(np.random.choice(len(exp_probs), p=exp_probs))

def generate_sequence(
    model: tf.keras.Model,
    seed: np.ndarray,
    int_to_note: dict,
    length: int,
    temperature: float,
) -> List[str]:

    context = list(seed.astype(int))
    generated: List[str] = []

    print(f"\nGenerating {length} tokens  (temperature={temperature}) ...")

    for step in range(length):
        x = np.array([context], dtype=np.int32)
        probs = model.predict(x, verbose=0)[0]
        idx = _sample(probs, temperature)
        generated.append(int_to_note[idx])

        context.append(idx)
        context.pop(0)

        if (step + 1) % 100 == 0:
            print(f"  {step + 1:>4} / {length} tokens generated")

    return generated

def sequence_to_midi(
    sequence: List[str],
    output_path: str,
    bpm: int = 120,
) -> str:
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_notes = []
    offset = 0.0

    for token in sequence:
        if "." in token:
            pitches = token.split(".")
            try:
                chord_notes = [note.Note(p) for p in pitches]
                new_chord   = chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            except Exception:
                pass
        else:
            try:
                new_note        = note.Note(token)
                new_note.offset = offset
                output_notes.append(new_note)
            except Exception:
                pass

        offset += 0.5

    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, tempo.MetronomeMark(number=bpm))
    for el in output_notes:
        part.append(el)

    score = stream.Score()
    score.append(part)
    score.write("midi", fp=output_path)

    print(f"\nMIDI saved → '{output_path}'")
    return output_path

def generate(
    model_path:  Optional[str] = None,
    output_path: Optional[str] = None,
    length:      Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:

    model_path  = model_path  or os.path.join(config.MODELS_DIR, "best_model.keras")
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "generated.mid")
    length      = length      or config.GENERATE_LENGTH
    temperature = temperature or config.TEMPERATURE

    print("Loading vocabulary ...")
    d = load_data(config.DATA_DIR, "elements", "note_to_int", "int_to_note")
    elements    = d["elements"]
    note_to_int = d["note_to_int"]
    int_to_note = d["int_to_note"]

    print(f"Vocabulary: {len(note_to_int)} tokens")

    X, _ = create_sequences(elements, note_to_int, config.SEQUENCE_LENGTH)

    seed_idx = np.random.randint(0, len(X))
    seed     = X[seed_idx]
    print(f"Seed index: {seed_idx}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Train first:  python main.py train"
        )
    print(f"Loading model from '{model_path}' ...")
    model = tf.keras.models.load_model(model_path)

    sequence = generate_sequence(
        model       = model,
        seed        = seed,
        int_to_note = int_to_note,
        length      = length,
        temperature = temperature,
    )

    sequence_to_midi(sequence, output_path, bpm=config.BPM)

    print("\nDone! Open the MIDI file in GarageBand, MuseScore, or any DAW.")
    return output_path

if __name__ == "__main__":
    generate()
