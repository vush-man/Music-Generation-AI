import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

import config
from data_preprocessing import (
    load_midi_dataset,
    build_vocabulary,
    create_sequences,
    save_data,
    load_data,
)
from model import build_model

def _header(title: str) -> None:
    width = 55
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _plot_history(history, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, color in zip(
        axes,
        ["loss", "accuracy"],
        ["tab:blue", "tab:green"],
    ):
        ax.plot(history.history[metric], color=color, lw=2)
        ax.set_title(f"Training {metric.capitalize()}", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.grid(alpha=0.3)

    plt.suptitle("Music Generation LSTM — Training History", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(out_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nTraining plot → '{path}'")

def train(force_preprocess: bool = False) -> tf.keras.Model:
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR,   exist_ok=True)

    elements_path = os.path.join(config.DATA_DIR, "elements.pkl")

    if force_preprocess or not os.path.exists(elements_path):
        _header("STEP 1 - Preprocessing MIDI files")
        elements = load_midi_dataset(config.MIDI_DIR)
        note_to_int, int_to_note = build_vocabulary(elements)
        save_data(
            config.DATA_DIR,
            elements=elements,
            note_to_int=note_to_int,
            int_to_note=int_to_note,
        )
    else:
        print("\nCached preprocessed data found — skipping MIDI parsing.")
        print("(Pass --force-preprocess to re-parse your dataset.)\n")
        d = load_data(config.DATA_DIR, "elements", "note_to_int", "int_to_note")
        elements    = d["elements"]
        note_to_int = d["note_to_int"]
        int_to_note = d["int_to_note"]

    n_vocab = len(note_to_int)
    print(f"\nVocabulary  : {n_vocab} unique tokens")
    print(f"Total notes : {len(elements):,}")

    _header("STEP 2 - Creating training sequences")
    X, y = create_sequences(elements, note_to_int, config.SEQUENCE_LENGTH)

    _header("STEP 3 - Building model")
    model = build_model(
        n_vocab       = n_vocab,
        seq_len       = config.SEQUENCE_LENGTH,
        embedding_dim = config.EMBEDDING_DIM,
        lstm_units    = config.LSTM_UNITS,
        dropout       = config.DROPOUT_RATE,
        learning_rate = config.LEARNING_RATE,
    )
    model.summary()

    best_ckpt  = os.path.join(config.MODELS_DIR, "best_model.keras")
    callbacks  = [
        ModelCheckpoint(
            best_ckpt,
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    _header("STEP 4 - Training")
    history = model.fit(
        X, y,
        epochs     = config.EPOCHS,
        batch_size = config.BATCH_SIZE,
        callbacks  = callbacks,
        verbose    = 1,
    )

    final_path = os.path.join(config.MODELS_DIR, "final_model.keras")
    model.save(final_path)
    print(f"\nFinal model → '{final_path}'")
    print(f"Best model  → '{best_ckpt}'")

    _plot_history(history, config.MODELS_DIR)
    print("\nTraining complete!")

    return model

if __name__ == "__main__":
    train()
