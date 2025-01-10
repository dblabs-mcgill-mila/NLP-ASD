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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import wandb

model_id = "meta-llama/Meta-Llama-3.1-8B"
HF_access = os.environ["HF_ACCESS"]

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Local path files
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
) # HP tuned for benchmarking

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

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=22)

for fold, (train_index, test_index) in enumerate(skf.split(tokenized_data, labels, groups=pat_ids)):

    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=HF_access, cache_dir=hf_dir, device_map="cuda", quantization_config=bnb_config)
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.hf_device_map[""] = model.device

    
    tokenized_train_dataset = dataset.select(train_index)
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_val_dataset = dataset.select(test_index)
    tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    wandb.init(project="NLP-ASD-llama-lora",
               name=f"Fold {fold}")

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            output_dir=hf_dir,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5, # Want about 10x smaller than the pre-trained learning rate
            logging_steps=214, #428
            #bf16=True,                #For TPU, set this to true
            fp16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=214,                # Save checkpoints every 50 steps
            eval_strategy="steps", # Evaluate the model every logging step
            eval_steps=214,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to='wandb',           # set to 'wandb' for weights & baises logging
            run_name=f"Fold {fold}",          
        ) # Will HP tune for benchmarking
    )

    trainer.train()
    trainer.evaluate()
    print(f"Finished training fold {fold}")

    wandb.finish()
