from config import torch, nn, F
from transformers_block import TransformerBlock
import nltk
import ssl

# Fix SSL certificate issue for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

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


# Create vocabulary and model
# Get unique words from Brown corpus and add special tokens
all_words = brown.words()
unique_words = list(set(all_words))[:1000]  # Limit vocabulary size
vocab = ["<unk>", "<pad>", "<mask>"] + unique_words  # Add special tokens
print(f"Vocab size: {len(vocab)}")
wordToId = {word: idx for idx, word in enumerate(vocab)}

def tokenizer(sentence, wordToId):
	words = sentence.split()
	tokens = []
	for word in words:
		if word in wordToId:
			tokens.append(wordToId[word])
		else:
			tokens.append(wordToId["<unk>"])
	return tokens

def detokenizer(tokens, vocab):
	return " ".join([vocab[token] for token in tokens])

input_tokens = torch.tensor([tokenizer("the cat ate <mask>", wordToId)])
print(f"Input tokens: {input_tokens}")

# Get model output
model = TinyTransformer(len(vocab), 128, 8, 2)
logits = model(input_tokens)
print(f"Logits shape: {logits.shape}")

# Get probabilities
probs = F.softmax(logits[0, -1, :], dim=-1)
print(f"Probabilities shape: {probs.shape}")

# Get predicted token IDs
predicted_indices = torch.argmax(probs, dim=-1)  # Shape: [1, 16]
print(f"Predicted indices: {predicted_indices}")

# Convert to words
predicted_words = detokenizer([predicted_indices.item()], vocab)
print(f"Predicted words: {predicted_words}")