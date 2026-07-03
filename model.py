import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.optimizers import Adam


def build_model(
    n_vocab: int,
    seq_len: int,
    embedding_dim: int = 64,
    lstm_units: int = 512,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
) -> tf.keras.Model:

    inputs = Input(shape=(seq_len,), name="note_sequence")

    x = Embedding(n_vocab, embedding_dim, name="embedding")(inputs)

    x = LSTM(lstm_units, return_sequences=True, name="lstm_1")(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    x = LSTM(lstm_units, return_sequences=False, name="lstm_2")(x)
    x = Dropout(dropout)(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation="relu", name="dense_1")(x)
    x = Dropout(dropout)(x)

    outputs = Dense(n_vocab, activation="softmax", name="output")(x)

    model = Model(inputs, outputs, name="MusicGenerationLSTM")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
