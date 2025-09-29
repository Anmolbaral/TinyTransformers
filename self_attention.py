from config import torch, nn

class SelfAttention(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()
		self.embed_dim = embed_dim
		self.query = nn.Linear(self.embed_dim, self.embed_dim)
		self.key = nn.Linear(self.embed_dim, self.embed_dim)
		self.value = nn.Linear(self.embed_dim, self.embed_dim)
	
	def forward(self, x):
		query = self.query(x)
		key = self.key(x)
		value = self.value(x)
		
		attentionScore = torch.matmul(query, key.transpose(-2, -1))/self.embed_dim**0.5
		weights = torch.softmax(attentionScore, dim=-1)
		output = torch.matmul(weights, value)
		return output







