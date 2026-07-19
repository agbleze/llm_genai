#%%
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, AutoModelForCausalLM
                          )
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# %%
def fine_tune_bert_for_sentiment_analysis():
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text = "The movie was great! I really enjoyed it."
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1)
    print(f"Predicted sentiment class: {predicted_class}")

#%%
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# %% fine tune gpt for text generation
def fine_tune_gpt_for_text_generation():
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    text = "write python function to read a csv file"
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(**inputs, max_length=1, num_return_sequences=1,
                            max_new_tokens=500
                            )
    generated_text = tokenizer.decode(outputs[0], 
                                      skip_special_tokens=True
                                      )
    print(f"Generated text: {generated_text}")



# %%

dataset = load_dataset("imdb", split="train[:1000]")

#%%
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#%%
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

#%%
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


#tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(output_dir="./results",
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  num_train_epochs=3,
                                  learning_rate=2e-5,
                                  weight_decay=0.01,
                                  )

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


#%%
# trainer = Trainer(model=model, args=training_args,
#                   train_dataset=tokenized_datasets,
#                   eval_dataset=tokenized_datasets,
#                   tokenizer=tokenizer,
#                   compute_metrics=compute_metrics
#                   )
# # %%
# trainer.train()
# # %%
# trainer.evaluate()
# %%
def fine_tune_and_evaluate(model_name="bert-base-uncased", dataset_name="imdb", 
                           num_labels=2, split="train[:1000]", 
                           max_length=128, batch_size=8, learning_rate=2e-5,
                           weight_decay=0.01,
                           num_train_epochs=3):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(dataset_name, split=split)
    
  
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(output_dir="./results",
                                      evaluation_strategy="epoch",
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_train_epochs,
                                      learning_rate=learning_rate,
                                      weight_decay=weight_decay,
                                      )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=tokenized_datasets,
                      eval_dataset=tokenized_datasets,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics
                      )
    
    trainer.train()
    evaluation_results = trainer.evaluate()
    return evaluation_results

#%%
accuracy_bert = fine_tune_and_evaluate(model_name="bert-base-uncased",
                                       dataset_name="imdb"
                                       )
# %%
print(f"Accuracy of BERT on IMDB: {accuracy_bert}")
# %%  L1 regularization
import torch.optim as optim

optimizer_l1 = optim.Adam(model.parameters(), lr=0.001)

lambda_l1 = 0.01  # Regularization strength

def l1_regularization(model, lambda_l1):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += torch.sum(torch.abs(param))
    return lambda_l1 * l1_norm


def train_step_l1(model, optimizer, criterion, data, target, lambda_l1):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target) + l1_regularization(model, lambda_l1)
    loss.backward()
    optimizer.step()
    return loss
# %% Mixed Precision Training

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# %%
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)
# %%
input_size = 10
output_size = 5
batch_size = 32
data = torch.randn(100, input_size)
target = torch.randn(100, output_size)
dataset = TensorDataset(data, target)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


#%%


def train_mixed_precision(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    model.train()
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            if (i + 1) % len(dataloader) == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

# %%
model = SimpleModel(input_size, output_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_mixed_precision(model, dataloader, criterion, optimizer, num_epochs=5, device=device)
# %%  distrobuted training: data parallelism
import torch.distributed as dist
import torch.multiprocessing as mp

# %%
def train_process(rank, world_size, dataset, batch_size, epochs, learning_rate):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_size = 10
    output_size = 5
    model = SimpleModel(input_size, output_size).to(rank)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    criterion = nn.MSELoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f"Rank: {rank} Epoch: {epoch+1}, Loss: {loss.item():.4f}")
            
    dist.destroy_process_group()
    

def run_distributed_training():
    """
    
    """
    world_size = 4
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    
    input_size = 10
    output_size = 5
    num_samples = 1000
    data = torch.randn(num_samples, input_size)
    target = torch.randn(num_samples, output_size)
    dataset = TensorDataset(data, target)
    
    mp.spawn(train_process, 
             args=(world_size, dataset, batch_size, epochs, learning_rate),
             nprocs=world_size,
             join=True
             )
    

# %%
run_distributed_training()
# %%
