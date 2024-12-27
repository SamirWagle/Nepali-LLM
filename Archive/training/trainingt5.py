import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random

# Load tokenizer and model
model_name = "t5-small"  # Replace with your specific T5 model variant if different
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the pad_token to eos_token if pad_token is missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Parameters
sequence_length = 128
batch_size = 32
num_epochs = 3
learning_rate = 8

# Sample dataset (use your actual Nepali text here)
text = "This is a sample text for training. Modify this with your actual Nepali text."
tokenized_text = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=sequence_length)["input_ids"].squeeze()
train_data = [tokenized_text[i:i + sequence_length] for i in range(0, len(tokenized_text) - sequence_length, sequence_length)]

# Function to create random batches
def get_random_batch(data, batch_size, sequence_length):
    if len(data) < batch_size:
        return None, None
    batch = random.sample(data, batch_size)
    input_batch = torch.stack([x[:-1] for x in batch])
    target_batch = torch.stack([x[1:] for x in batch])
    return input_batch, target_batch

# Training loop
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

if len(train_data) >= batch_size:
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = len(train_data) // batch_size
        for i in range(batch_count):
            input_batch, target_batch = get_random_batch(train_data, batch_size, sequence_length)
            if input_batch is None or target_batch is None:
                continue

            optimizer.zero_grad()
            outputs = model(input_ids=input_batch, labels=target_batch)
            loss = outputs.loss
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
    input_ids = tokenizer("question: " + question, return_tensors="pt", truncation=True)["input_ids"]
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, eos_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# Test the model with a question
question = "नेपालको राजधानी कहाँ हो?"
print("Question:", question)
answer = answer_question(question)
print("Answer:", answer)
