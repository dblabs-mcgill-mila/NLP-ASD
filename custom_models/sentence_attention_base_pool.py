import numpy as np

import torch
from torch import nn

from transformers import AutoModel, AutoModelForSequenceClassification

class SelfAttentionAverage(nn.Module):
    def __init__(self, embed_dim=768, weight_dim=128, dropout=0.0, classifier_dropout=0.0):
        super(SelfAttentionAverage, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads=1, 
            kdim=weight_dim, 
            vdim=weight_dim,
            batch_first=True,
            dropout=dropout 
        )

        self.dropout_layer = nn.Dropout(p=classifier_dropout)

        self.linear = nn.Linear(weight_dim, 1)
    
    def forward(self, x, attn_mask=None, avgerage_attn_weights=False):
        attn_output, attn_weights = self.attn(
            query=x, 
            key=x, 
            value=x, 
            attn_mask=attn_mask,
            average_attn_weights=avgerage_attn_weights
        )

        pooled_output = torch.mean(attn_output, dim=1)
        pooled_output_drop = self.dropout_layer(pooled_output)

        log_reg = self.linear(pooled_output_drop).squeeze()

        
        return log_reg, attn_weights, attn_output
    

class SentenceAttentionBERT(nn.Module):
    def __init__(
        self, 
        base_model_name, 
        sentence_embed_dim=768, 
        sentence_weight_dim=768,
        word_embed_dim=768,
        word_weight_dim=768,
        sentence_dropout=0.0,
        ff_dropout=0.1,
        att_dropout=0.1,
        report_max_length=64,
        class_dropout=0.0
    ):
        super(SentenceAttentionBERT, self).__init__()

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, 
            summary_use_proj=False,
            summary_proj_to_labels=False,
            summary_type="mean",
            dropout=ff_dropout,
            attention_dropout=att_dropout
        ) # change dropouts
        self.sentence_attention = SelfAttentionAverage(
            embed_dim=sentence_embed_dim,
            weight_dim=sentence_weight_dim,
            dropout=sentence_dropout,
            classifier_dropout=class_dropout
        )
        
        self.word_attention = nn.MultiheadAttention(
            word_embed_dim, 
            num_heads=6, 
            kdim=word_weight_dim, 
            vdim=word_weight_dim,
            batch_first=True,
            dropout=sentence_dropout 
        )
        
        self.report_max_length = report_max_length
        
    
    def forward(self, input, attn_mask=None, sentence_attn_mask=None):
        # input has shape (N, S, T)
        # N: batch size
        # S: number of sentences
        # T: number of tokens (words)

        batch_cls = []
        word_attentions_batch = []
        for report_ind in range(input.shape[0]):
            report_cls = []
            word_attentions_report = []
            for micro_batch_inds in range(int(input.shape[1] / self.report_max_length)):
                single_report_input = input[report_ind, 
                                            micro_batch_inds*self.report_max_length:(micro_batch_inds+1)*self.report_max_length,
                                            :].squeeze()
                single_report_mask = attn_mask[report_ind, 
                                               micro_batch_inds*self.report_max_length:(micro_batch_inds+1)*self.report_max_length,
                                               :].squeeze()
                base_outputs = self.base_model(
                    single_report_input,
                    attention_mask = single_report_mask
                )
                base_cls = base_outputs.logits
                report_cls.append(base_cls)
            report_cls = torch.cat(report_cls, dim=0)
            batch_cls.append(report_cls)
        batch_cls = torch.stack(batch_cls, dim=0)
        
        logits, sentence_attn_weights, attn_output = self.sentence_attention(batch_cls, attn_mask=sentence_attn_mask)
        
        return logits, sentence_attn_weights, attn_output
