
#%%
import torch
import torch.nn as nn


# %%
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        hidden = self.linear(embedded)
        output = self.output_layer(hidden)
        return output
    
    
#%%

vocab_size = 10_000
embedding_dim = 256
hidden_dim = 256
seq_len = 32
batch_size = 64


model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_dim)
input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))

output_tensor = model(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)
# %%
