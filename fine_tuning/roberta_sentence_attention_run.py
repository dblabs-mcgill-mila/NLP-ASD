### Imports ###

import os
import numpy as np
import pandas as pd

import torch
from torch import nn

from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold

from helpers.batcher import create_batches
from custom_models.sentence_attention_base_pool import SentenceAttentionBERT

import wandb
import argparse

### Set up argparse ###
parser = argparse.ArgumentParser(description='Training Run')

parser.add_argument("--epochs", default=4, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--optimizer", default="AdamW", type=str)
parser.add_argument("--ff_dropout", default=0.1, type=float)
parser.add_argument("--att_dropout", default=0.1, type=float)
parser.add_argument("--class_dropout", default=0.1, type=float)
parser.add_argument("--sentence_dropout", default=0.0, type=float)
parser.add_argument("--weight_decay", default=0.01, type=float)
parser.add_argument("--job_id", default="0", type=str)
parser.add_argument("--k_fold_CV", default=5, type=int)

args = parser.parse_args()

# Local paths
ROOT = "../"
DATA_PATH = os.path.join(ROOT, "data")
REPORT_PATH = os.path.join(DATA_PATH, "reports_txt")

# Load tokenied data
input_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "input_tensor"))
attention_mask_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "attention_mask_tensor"))
label_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "label_tensor"))
report_id_array = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "report_id_array"))

pat_ids = []
for report_id in report_id_array:
    pat_ids.append(report_id[:7])

### Wandb Logging Function ###
    
class Evaluator():
    def __init__(self, batch_size, n_train_data, n_valid_data, eval_strat = "epoch", eval_freq = 2):
        self.batch_size = batch_size
        self.n_train_data = n_train_data
        self.n_train_batches = np.ceil(n_train_data / batch_size)
        self.n_valid_data = n_valid_data
        self.n_valid_batches = np.ceil(n_valid_data / batch_size)
        
        self.eval_strat = eval_strat
        self.eval_freq = eval_freq
        
        self.run_valid = False
    
    def collect(self, 
                train_batch_step=None,
                train_loss=None,
                train_probs=None,
                train_true_labels=None,
                valid_batch_step=None,
                valid_loss=None,
                valid_probs=None,
                valid_true_labels=None,
                collect_type="train"
               ):
        
        
        if collect_type == "train":
            self.train_batch_step = train_batch_step
            # Restart collection at first batch of each epoch
            if self.train_batch_step == 0:
                self.metric_dict = {
                    "train_loss": [],
                    "train_probs": [],
                    "train_true_labels": [],
                    "valid_loss": [],
                    "valid_probs": [],
                    "valid_true_labels": [],
                }
            self.metric_dict["train_loss"].append(train_loss.cpu().detach().numpy())
            self.metric_dict["train_probs"].extend(torch.atleast_1d(train_probs).cpu().detach().numpy())
            self.metric_dict["train_true_labels"].extend(torch.atleast_1d(train_true_labels).cpu().detach().numpy())
        
        
        if collect_type == "valid":
            self.valid_batch_step = valid_batch_step
            # Restart valid collection every time we evaluate (might be more frequent than epoch)
            if self.valid_batch_step == 0:
                self.metric_dict["valid_loss"] = []
                self.metric_dict["valid_probs"] = []
                self.metric_dict["valid_true_labels"] = []
            self.metric_dict["valid_loss"].append(valid_loss.cpu().detach().numpy())
            self.metric_dict["valid_probs"].extend(torch.atleast_1d(valid_probs).cpu().detach().numpy())
            self.metric_dict["valid_true_labels"].extend(torch.atleast_1d(valid_true_labels).cpu().detach().numpy())
        
        if (self.eval_strat == "mid_epoch" and self.train_batch_step % np.ceil(self.n_train_batches / self.eval_freq) == 0
            or self.train_batch_step == self.n_train_batches - 1):
            
            self.run_valid = True

        else:
            self.run_valid = False
    
    def accuracy_score(self, probs, targets):
        return sum(round(p) == t for p, t in zip(probs, targets)) / len(targets)
    
    def evaluate(self):
        # Always evalute train and valid at the end of epoch
        total_valid_loss = np.sum(self.metric_dict["valid_loss"]) / self.n_valid_data
        total_valid_acc = self.accuracy_score(self.metric_dict["valid_probs"], self.metric_dict["valid_true_labels"])

        wandb.log({
            "valid_loss": total_valid_loss,
            "valid_acc": total_valid_acc,
        })
            
        if self.train_batch_step == self.n_train_batches - 1:
            total_train_loss = np.sum(self.metric_dict["train_loss"]) / self.n_train_data
            total_train_acc = self.accuracy_score(self.metric_dict["train_probs"], self.metric_dict["train_true_labels"])
            total_valid_loss = np.sum(self.metric_dict["valid_loss"]) / self.n_valid_data
            total_valid_acc = self.accuracy_score(self.metric_dict["valid_probs"], self.metric_dict["valid_true_labels"])
            
            wandb.log({
                "train_loss": total_train_loss,
                "train_acc": total_train_acc,
            })
            
            return total_train_loss, total_train_acc, total_valid_loss, total_valid_acc        
        return None

### Training Loop ###
    
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
device = "cuda"

skf = StratifiedGroupKFold(n_splits=args.k_fold_CV, shuffle=True, random_state=22)
splits = skf.split(input_tensor, label_tensor, groups=pat_ids)

input_tensor = input_tensor.to(device)
attention_mask_tensor = attention_mask_tensor.to(device)
label_tensor = label_tensor.float().to(device)

for index, (train_idx, valid_idx) in enumerate(splits):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="NLP-ASD",
        name=f"HP_CV_{args.job_id}",
        group=f"Fold {index+1}",
        # track hyperparameters and run metadata
        config={
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": args.optimizer,
                "ff_dropout": args.ff_dropout,
                "att_dropout": args.att_dropout,
                "class_dropout": args.class_dropout,
                "sentence_dropout": args.sentence_dropout,
                "weight_decay": args.weight_decay,
            }
    )
    
    train_data = create_batches(input_tensor[train_idx], batch_size)
    train_mask = create_batches(attention_mask_tensor[train_idx], batch_size)
    valid_data = create_batches(input_tensor[valid_idx], batch_size)
    valid_mask = create_batches(attention_mask_tensor[valid_idx], batch_size)
    
    train_labels = create_batches(label_tensor[train_idx], batch_size)
    valid_labels = create_batches(label_tensor[valid_idx], batch_size)
    
    model = SentenceAttentionBERT("flaubert/flaubert_base_cased", report_max_length=64, ff_dropout=args.ff_dropout , att_dropout=args.att_dropout, class_dropout=args.class_dropout, sentence_dropout=args.sentence_dropout)
    model = nn.DataParallel(model)
    model = model.to(device)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    
    ev = Evaluator(
        batch_size = batch_size, 
        n_train_data = len(train_idx), 
        n_valid_data = len(valid_idx), 
        eval_strat = "mid_epoch",
        eval_freq = 3
    )
    
    for epoch in range(epochs):
        
        for train_batch in range(len(train_data)):
            model.zero_grad()
            
            output, _, _ = model(train_data[train_batch], attn_mask=train_mask[train_batch])
            output = torch.atleast_1d(output)
            train_labels_true = torch.atleast_1d(train_labels[train_batch])
            loss = loss_fn(output, train_labels_true)

            ev.collect(
                train_batch_step=train_batch,
                train_loss=loss,
                train_probs=torch.sigmoid(output),
                train_true_labels=train_labels_true,
                collect_type="train"
            )
            
            loss.backward()
            optimizer.step()

            
            if ev.run_valid:
                model.eval()
                with torch.no_grad():
                    for valid_batch in range(len(valid_data)):
                        output, _, _ = model(valid_data[valid_batch], attn_mask=valid_mask[valid_batch])
                        output = torch.atleast_1d(output)
                        valid_labels_true = torch.atleast_1d(valid_labels[valid_batch])
                        loss = loss_fn(output, valid_labels_true)

                        ev.collect(
                            train_batch_step = ev.train_batch_step,
                            
                            valid_batch_step=valid_batch,
                            valid_loss=loss,
                            valid_probs=torch.sigmoid(output),
                            valid_true_labels=valid_labels_true,
                            collect_type = "valid"
                        )
                    ev.evaluate()

        print("Epoch Done")

    wandb.finish()
