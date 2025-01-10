import os
hf_dir = os.path.join(os.environ['SCRATCH'], "hugging_face")
os.environ['HF_HOME'] = hf_dir

import re
import math
import numpy as np
import pandas as pd
from cleantext import clean

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import Dataset
from peft import prepare_model_for_kbit_training,LoraConfig,PeftModel,get_peft_model

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import wandb

model_id = "google/gemma-7b"
HF_access = os.environ["HF_ACCESS"]

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Local path to files
ROOT = "../"
DATA_PATH = os.path.join(ROOT, "data")
REPORT_PATH = os.path.join(DATA_PATH, "reports_txt")
    
data_df = pd.read_csv(
    os.path.join(REPORT_PATH, "raw_text_df.csv")
)
data_df.columns = ["text", "labels", "pat_id"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
     #bnb_4bit_compute_dtype="float16",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_access, cache_dir=hf_dir, pad_token='<pad>')
model = AutoModelForSequenceClassification.from_pretrained(model_id, token=HF_access, cache_dir=hf_dir, device_map="cuda", quantization_config=bnb_config)

model.resize_token_embeddings(len(tokenizer))

## Tokenize the text
def tokenize_text(text):
    return tokenizer(text, truncation=True, padding="max_length", max_length=4096, return_tensors="pt")
ad_token=" "
tokenized_data = []
labels = []
pat_ids = []
for idx, row in data_df.iterrows():
    tokenized_data.append(tokenize_text(row.text))
    labels.append(row.labels)
    pat_ids.append(row.pat_id)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r = 8,
    lora_alpha = 8,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias = "none",
    lora_dropout=0.1,  # Conventional
    task_type = "SEQ_CLS"
) # Will HP tune for benchmarking

model = get_peft_model(model, config)
model.config.pad_token_id = tokenizer.pad_token_id

data_dict = {
    "input_ids": torch.stack([d["input_ids"] for d in tokenized_data]).squeeze(),
    "attention_mask": torch.stack([d["attention_mask"] for d in tokenized_data]).squeeze(),
    "labels": torch.tensor(labels)
}
dataset = Dataset.from_dict(data_dict)

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

model.hf_device_map[""] = model.device

outputs = []
for i in range(len(tokenized_data)):
    input_ids = tokenized_data[i]['input_ids'].to('cuda')
    attention_mask = tokenized_data[i]['attention_mask'].to('cuda')
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    outputs.append(output.last_hidden_state[0,:,:].cpu().numpy().mean(axis=1))
    if i % 100 == 0:
        print(f"Processed {i} reports")

embeddings = np.array(outputs)

# Predict the labels from the embeddings using log reg.

import wandb

X = embeddings
y = np.array(labels)
pat_ids = np.array(pat_ids)

skf = StratifiedGroupKFold(n_splits=5, random_state=22)

for i, (train_index, test_index) in enumerate(skf.split(X, y, groups=pat_ids)):

    wandb.init(
        project="NLP-ASD-gemma-zs",
        name=f"fold_{i}"
    )

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = LogisticRegressionCV(max_iter=1000) # HP tune for full benchmarking
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(clf.score(X_test, y_test))

    wandb.log({"test accuracy": acc})
    wandb.log({"test f1": f1})

wandb.finish()