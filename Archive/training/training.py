import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load("C:/Users/L E G I O N/Desktop/Nepali-LLM/data/your_model.model")

# Parameters
vocab_size = sp.get_piece_size()
embed_dim = 256
num_layers = 2
num_heads = 4
sequence_length = 128
batch_size = 1  # Reduced batch size to work with limited data
num_epochs = 3
learning_rate = 1e-4

# Define a simple GPT-like model
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super(SimpleGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(embed_dim, num_heads, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)  # Transformer expects [sequence_length, batch_size, embed_dim]
        x = self.transformer(x, x)
        return self.fc(x).permute(1, 0, 2)  # Back to [batch_size, sequence_length, vocab_size]

# Initialize model, optimizer, and loss function
model = SimpleGPT(vocab_size, embed_dim, num_layers, num_heads)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Prepare the dataset (example with dummy text)

text = "This is some sample text for training the language model. This model will try to predict the next words."
token_ids = sp.encode(text, out_type=int)
train_data = [token_ids[i:i + sequence_length] for i in range(0, len(token_ids) - sequence_length, sequence_length)]

# Function to create random batches
def get_random_batch(data, batch_size, sequence_length):
    if len(data) < batch_size:
        return None, None  # Not enough data for the batch
    batch = random.sample(data, batch_size)
    input_batch = torch.tensor([x[:-1] for x in batch], dtype=torch.long)
    target_batch = torch.tensor([x[1:] for x in batch], dtype=torch.long)
    return input_batch, target_batch

# Training loop
if len(train_data) >= batch_size:
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = len(train_data) // batch_size
        for i in range(batch_count):
            input_batch, target_batch = get_random_batch(train_data, batch_size, sequence_length)
            if input_batch is None or target_batch is None:
                continue  # Skip if batch data is insufficient

            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output.view(-1, vocab_size), target_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
else:
    print("Not enough data to proceed with training. Try a smaller batch size or add more data.")

# Function for answering questions
def answer_question(question, max_length=50):
    model.eval()
    tokens = sp.encode(question, out_type=int)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_ids)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            tokens.append(next_token)
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            if next_token == sp.piece_to_id("</s>"):  # Stop if end of sentence
                break
    
    answer = sp.decode(tokens)
    return answer

# Test the model with a question
question = "नेपालको राजधानी कहाँ हो?"
print("Question:", question)
answer = answer_question(question)
print("Answer:", answer)
