from config import torch, nn, F
from transformers_block import TransformerBlock
import nltk

# Download brown corpus if not already available
try:
    from nltk.corpus import brown
    brown.words()
except LookupError:
    print("Downloading Brown corpus...")
    nltk.download('brown')
    from nltk.corpus import brown


class TinyTransformer(nn.Module):
	def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.transformerBlocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
		self.fc = nn.Linear(embed_dim, vocab_size)

	def forward(self, x):
		x = self.embedding(x)
		for block in self.transformerBlocks:
			x = block(x)
		x = self.fc(x)
		return x


# Create vocabulary
all_words = brown.words()
unique_words = list(set(all_words))[:1000]  # Limit vocabulary size
vocab = ["<unk>", "<pad>", "<mask>"] + unique_words  # Add special tokens
wordToId = {word: idx for idx, word in enumerate(vocab)}

def tokenizer(sentence, wordToId):
	words = sentence.lower().split()
	tokens = []
	for word in words:
		if word in wordToId:
			tokens.append(wordToId[word])
		else:
			tokens.append(wordToId["<unk>"])
	return tokens

# Example: next-word prediction for a context (no <mask>)
context = "the cat ate"
input_tokens = torch.tensor([tokenizer(context, wordToId)])  # [1, seq_len]
print(f"Context: '{context}' → tokens: {input_tokens}")

# Build model and switch to eval for inference
model = TinyTransformer(len(vocab), 128, 8, 2)
model.eval()

with torch.no_grad():
    logits = model(input_tokens)  # [1, seq_len, vocab]
    next_token_logits = logits[0, -1, :]  # logits for next word given context
    probs = F.softmax(next_token_logits, dim=-1)
    predicted_id = torch.argmax(probs).item()

predicted_word = vocab[predicted_id]
print(f"Predicted next word: {predicted_word}")