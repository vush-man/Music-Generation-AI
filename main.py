import argparse
import sys

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Music Generation AI — LSTM-based MIDI composer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser(
        "preprocess",
        help="Parse MIDI files and build the vocabulary cache.",
    )
    pre.add_argument(
        "--midi-dir",
        default=None,
        help="Override the MIDI source directory from config.py.",
    )

    trn = sub.add_parser(
        "train",
        help="Train the LSTM model.",
    )
    trn.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Re-parse MIDI files even when cached data already exists.",
    )

    gen = sub.add_parser(
        "generate",
        help="Generate a new MIDI composition using a trained model.",
    )
    gen.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to a .keras model file  (default: models/best_model.keras).",
    )
    gen.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output MIDI path  (default: output/generated.mid).",
    )
    gen.add_argument(
        "--length",
        type=int,
        default=None,
        metavar="N",
        help="Number of notes / chords to generate  (default: config.GENERATE_LENGTH).",
    )
    gen.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="T",
        help=(
            "Sampling temperature — controls creativity:\n"
            "  0.5  very conservative\n"
            "  1.0  balanced (default)\n"
            "  1.5  very experimental"
        ),
    )

    return parser

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "preprocess":
        import config
        from data_preprocessing import (
            load_midi_dataset, build_vocabulary, save_data
        )
        midi_dir = args.midi_dir or config.MIDI_DIR
        elements = load_midi_dataset(midi_dir)
        note_to_int, int_to_note = build_vocabulary(elements)
        save_data(
            config.DATA_DIR,
            elements    = elements,
            note_to_int = note_to_int,
            int_to_note = int_to_note,
        )
        n_vocab = len(note_to_int)
        print(f"\nPreprocessing complete ✓")
        print(f"Vocabulary size : {n_vocab}")
        print(f"Total elements  : {len(elements):,}")
        print(f"\nNext step:  python main.py train")

    elif args.command == "train":
        from train import train
        train(force_preprocess=args.force_preprocess)
        print("\nNext step:  python main.py generate")

    elif args.command == "generate":
        from generate import generate
        output = generate(
            model_path  = args.model,
            output_path = args.output,
            length      = args.length,
            temperature = args.temperature,
        )
        print(f"\nGenerated MIDI: {output}")

if __name__ == "__main__":
    main()
