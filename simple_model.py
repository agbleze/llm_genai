
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

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        output, hidden = self.lstm(embedded, hidden)
        output = self.output_layer(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size),
                weight.new_zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
                )
        
        
vocab_size = 10_000
embedding_dim = 256
hidden_dim = 512
num_layers = 2
seq_len = 32
batch_size = 64


model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))
output_tensor, _ = model(input_tensor)
print(output_tensor.shape)


# %% loss
predictions = torch.randn(batch_size, seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (batch_size, seq_len))

loss_function = nn.CrossEntropyLoss()

loss = loss_function(predictions.view(-1, vocab_size), targets.view(-1))

print(f"Loss: {loss.item()}")
# %%
targets.view(-1).shape


predictions.view(-1, vocab_size).shape, targets.view(-1).shape
#%%
predictions.view(-1, vocab_size).shape
#%%

targets.shape
# %%
predictions[0][0].shape
# %% optimizer

import torch.optim as optim

model = nn.Linear(10,5)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# %%
optimizer
# %%
