from config import torch, nn, F
from multi_head_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
	def __init__(self, embed_dim, num_heads):
		super().__init__()
		self.embed_dim = embed_dim
		self.ff = nn.Sequential(
			nn.Linear(embed_dim, embed_dim*4),
			nn.ReLU(),
			nn.Linear(embed_dim*4, embed_dim)
		)
		self.attention = MultiHeadAttention(embed_dim, num_heads)
		self.norm2 = nn.LayerNorm(embed_dim)

	def forward(self, x):
		#Residual + norm after attention
		attnOut = self.attention(x)
		x = self.norm2(x + attnOut)

		#Residual + norm after feedforward
		ffOut = self.ff(x)
		x = self.norm2(x + ffOut)
		
		return x
