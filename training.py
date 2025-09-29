from config import torch, nn, F, optim, brown
from model import TinyTransformer
from model import wordToId
from model import vocab


model = TinyTransformer(len(vocab), 128, 8, 2)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

for epoch in range(EPOCHS):
	totalLoss = 0
	for sentence in brown.sents()[:500]:
		tokens = [wordToId.get(word.lower(), wordToId["<mask>"]) for word in sentence]
		if len(tokens) < 2:
			continue
		
		input = torch.tensor(tokens[:-1])
		target = torch.tensor(tokens[1:])

		logits = model(input)

		logits = logits.view(-1, logits.size(-1))
		loss = criterion(logits, target)
		
		#Back Propagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
		totalLoss += loss.item()
	
	print(f"Epoch {epoch+1}, Loss: {totalLoss:.4f}")




