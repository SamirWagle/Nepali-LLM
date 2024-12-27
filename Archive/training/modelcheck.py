import sentencepiece as spm

sp = spm.SentencePieceProcessor()
if sp.load("C:/Users/L E G I O N/Desktop/Nepali-LLM/data/your_model.model"):
    print("Model loaded successfully!")
else:
    print("Error loading model!")
