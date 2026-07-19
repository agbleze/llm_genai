
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# %% scaled dot-product attention

import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

batch_size = 32
num_heads = 8
seq_len = 10
d_k = 64
d_v = 64


query = torch.randn(batch_size, num_heads, seq_len, d_k)
key = torch.randn(batch_size, num_heads, seq_len, d_k)
value = torch.randn(batch_size, num_heads, seq_len, d_k)

output, attention_weights = scaled_dot_product_attention(query, key, value)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
        
    

# %%

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
    
    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_weights
    
    
batch_size, seq_len, d_model, num_heads = 32, 10, 512, 8
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)


attention = MultiHeadAttention(d_model, num_heads)
output, attn_weights = attention(Q, K, V)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attn_weights.shape}")


# %% prepare data for Masked Language Modeling (MLM)
from transformers import BertTokenizer, Auto


# %%

def create_masked_lm_dataset(texts, tokenizer, mlm_probability=0.15,
                             max_length=128
                             ):
    input_ids = []
    attention_mask = []
    labels = []
    
    for text in texts:
        encoded_text = tokenizer.encode_plus(text, add_special_tokens=True,
                                             max_length=max_length,
                                             padding="max_length",
                                             truncation=True,
                                             return_tensors="pt"
                                             )
        input_ids_tensor = encoded_text["input_ids"].squeeze(0)
        attention_mask_tensor = encoded_text["attention_mask"].squeeze(0)
        labels_tensor = input_ids_tensor.clone()
        
        rand = torch.rand(input_ids_tensor.shape)
        mask_arr = (rand * mlm_probability).floor().bool()
        masked_input_ids = input_ids_tensor.clone()
        masked_input_ids[mask_arr] = tokenizer.mask_token_id
        print(f"tokenizer.mask_token_id: {tokenizer.mask_token_id}")
        
        input_ids.append(masked_input_ids)
        attention_mask.append(attention_mask_tensor)
        labels.append(labels_tensor)
        
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = torch.stack(labels)
        return input_ids, attention_mask, labels

#%%    
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
texts = ["The cat sat on the mat.", "A quick brown fox jumps over the lazy dog."]

#%%
input_ids, attention_mask, labels = create_masked_lm_dataset(texts, tokenizer)

#%%
print(f"Input IDs shape: {input_ids.shape}")
print(f"Attention Maks shape: {attention_mask.shape}")
print(f"Labels shape: {labels.shape}")
# %%
encoded_text = tokenizer.encode_plus(texts[0], 
                                     add_special_tokens=True,
                                    max_length=128,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt"
                                    )
# %%
encoded_text.keys()
# %%
input_ids_tensor = encoded_text["input_ids"].squeeze(0)
# %%
rand = torch.rand(input_ids_tensor.shape)
# %%
rand.shape
# %%
mask_arr = (rand * 0.15).floor().bool()
# %%
mask_arr.shape
# %%
