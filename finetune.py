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
# %%
