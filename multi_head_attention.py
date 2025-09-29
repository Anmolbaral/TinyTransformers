from self_attention import SelfAttention
from config import torch, nn

class MultiHeadAttention(nn.Module):
	def __init__(self, query_dim, num_heads):
		super().__init__()
		self.heads = nn.ModuleList([SelfAttention(query_dim) for _ in range(num_heads)])
		self.linear = nn.Linear(query_dim * num_heads, query_dim)

	def forward(self, x):
		heads = [head(x) for head in self.heads]
		output = self.linear(torch.cat(heads, dim=-1))
		return output


