# TinyTransformers

A minimal implementation of the Transformer architecture from scratch using PyTorch. This project demonstrates the core components of transformers including self-attention, multi-head attention, and transformer blocks.

## 🚀 Features

- **Self-Attention Mechanism**: Implements scaled dot-product attention
- **Multi-Head Attention**: Parallel attention heads for capturing different types of relationships
- **Transformer Blocks**: Complete transformer blocks with residual connections and layer normalization
- **TinyTransformer Model**: A complete transformer model for next-word prediction
- **Training Pipeline**: Includes training script with the Brown corpus dataset
- **Vocabulary Management**: Tokenization and vocabulary handling with special tokens

## 📁 Project Structure

```
TinyTransformers/
├── config.py              # Configuration and imports
├── self_attention.py      # Self-attention implementation
├── multi_head_attention.py # Multi-head attention mechanism
├── transformers_block.py  # Transformer block with feed-forward network
├── model.py              # Main TinyTransformer model
├── training.py           # Training script
├── main.py              # Example usage and inference
└── requirements.txt     # Dependencies
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TinyTransformers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for Brown corpus):
```python
import nltk
nltk.download('brown')
```

## 🎯 Usage

### Basic Inference
```python
from model import TinyTransformer, tokenizer, wordToId, vocab
import torch

# Create model
model = TinyTransformer(vocab_size=len(vocab), embed_dim=128, num_heads=8, num_layers=2)
model.eval()

# Predict next word
context = "the cat ate"
input_tokens = torch.tensor([tokenizer(context, wordToId)])
with torch.no_grad():
    logits = model(input_tokens)
    # Get prediction...
```

### Training
```python
python training.py
```

## 🏗️ Architecture

### Self-Attention
- Scaled dot-product attention mechanism
- Query, Key, Value projections
- Attention weights computation

### Multi-Head Attention
- Parallel attention heads
- Concatenation and linear projection
- Captures different types of relationships

### Transformer Block
- Multi-head attention with residual connection
- Feed-forward network (4x expansion)
- Layer normalization after each sub-layer

### TinyTransformer Model
- Embedding layer
- Stack of transformer blocks
- Final linear projection to vocabulary size

## 📊 Model Configuration

- **Vocabulary Size**: 1000 words (from Brown corpus)
- **Embedding Dimension**: 128
- **Number of Heads**: 8
- **Number of Layers**: 2
- **Feed-forward Dimension**: 512 (4x embedding)

## 🎓 Learning Objectives

This implementation helps understand:
- How attention mechanisms work
- The role of residual connections and layer normalization
- Multi-head attention and its benefits
- Complete transformer architecture
- Training transformers for language modeling

## 📚 Dependencies

- PyTorch
- NumPy
- Pandas
- NLTK


## 📄 License

This project is open source and available under the [MIT License](LICENSE).
