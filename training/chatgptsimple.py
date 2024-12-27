import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load your preprocessed data
# Assuming `sequences` is a list of tokenized sequences
# `word_index` is a dictionary mapping words to their indices
# `embedding_matrix` is a pre-trained embedding matrix
# `vocab_size` is the size of your vocabulary
# Replace these placeholders with your actual data
sequences = [...]  # Example: [[1, 2, 3], [2, 3, 4], ...]
word_index = {...}  # Example: {'<PAD>': 0, 'word1': 1, 'word2': 2, ...}
embedding_matrix = np.load("embedding_matrix.npy")  # Load your embedding matrix
vocab_size = len(word_index) + 1
embedding_dim = embedding_matrix.shape[1]

# Prepare data for training
# Input sequences: X, and target words: y
X, y = [], []
for seq in sequences:
    X.append(seq[:-1])  # All words except the last
    y.append(seq[-1])   # The last word as the target

X = pad_sequences(X, padding="post")
y = to_categorical(y, num_classes=vocab_size)

# Build the text generation model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
              weights=[embedding_matrix], trainable=False, input_length=X.shape[1]),
    LSTM(128, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X, y, epochs=20, batch_size=64)

# Function to generate text
def generate_text(seed_text, num_words, max_seq_length):
    for _ in range(num_words):
        tokenized = [word_index.get(word, 0) for word in seed_text.split()]
        tokenized = pad_sequences([tokenized], maxlen=max_seq_length, padding="post")
        predicted = np.argmax(model.predict(tokenized), axis=-1)
        output_word = next((word for word, index in word_index.items() if index == predicted[0]), None)
        if output_word:
            seed_text += " " + output_word
        else:
            break
    return seed_text

# Example usage
seed_text = "नेपाल"
generated_text = generate_text(seed_text, num_words=50, max_seq_length=X.shape[1])
print(generated_text)
