import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------
# 1. Load and Preprocess MIDI Files
# ----------------------------------------------------
notes = []

print("Parsing MIDI files...")
for file in glob.glob("midi_songs/*.mid*"):
    midi = converter.parse(file)
    notes_to_parse = None

    try:
        # Group by instrument parts
        parts = instrument.partitionByInstrument(midi)
        notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
    except Exception:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            # Encode chords as string of dot-separated integers
            notes.append(".".join(str(n) for n in element.normalOrder))

# Save notes mapping for generation phase
with open("notes.pkl", "wb") as filepath:
    pickle.dump(notes, filepath)

# ----------------------------------------------------
# 2. Prepare Sequences for LSTM
# ----------------------------------------------------
sequence_length = 100
pitch_names = sorted(set(item for item in notes))
n_vocab = len(pitch_names)

# Map pitches to integers
note_to_int = {note_name: num for num, note_name in enumerate(pitch_names)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

# Reshape & normalize input for LSTM: (samples, time_steps, features)
X = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
y = to_categorical(network_output, num_classes=n_vocab)

# ----------------------------------------------------
# 3. Build & Train the LSTM Model
# ----------------------------------------------------
model = Sequential([
    LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(n_vocab, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# Train model (adjust epochs based on dataset size/hardware)
model.fit(X, y, epochs=50, batch_size=64)
model.save("music_generator_model.keras")
print("Training complete! Model saved as music_generator_model.keras")
