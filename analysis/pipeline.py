### Imports ###

import os
import numpy as np
import pandas as pd
from string import punctuation

import torch
import torch.nn.functional as F


from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


from helpers.hierarchical_tokenizer import hierarchical_tokenizer
from helpers.batcher import create_batches
from custom_models.sentence_attention_base_pool import SentenceAttentionBERT

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter



import ptitprince as pt
from adjustText import adjust_text

def np_sigmoid(x):
    return 1/(1 + np.exp(-x))

### Load Data ###
ROOT = "../"
DATA_PATH = os.path.join(ROOT, "data")
REPORT_PATH = os.path.join(DATA_PATH, "reports_txt")

input_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "input_tensor"), map_location=torch.device('cpu'))
attention_mask_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "attention_mask_tensor"), map_location=torch.device('cpu'))
label_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "label_tensor"), map_location=torch.device('cpu'))
report_id_array = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "report_id_array"), map_location=torch.device('cpu'))

pat_ids = []
for report_id in report_id_array:
    pat_ids.append(report_id[:7])

### Load Model ###
model = SentenceAttentionBERT("flaubert/flaubert_base_cased", report_max_length=64)
model.load_state_dict(torch.load(os.path.join(ROOT, "trained_models", "sentence_roberta_final.pt"), map_location=torch.device('cpu')))

# Model internals
logits_np = np.load(os.path.join(DATA_PATH, "intermediates", "FINAL_RESULTS", "logits_np.npy"))
attention_matrices_np = np.load(os.path.join(DATA_PATH, "intermediates", "FINAL_RESULTS", "attention_matrices_np.npy"))
attention_weighted_sentence_embs_np = np.load(os.path.join(DATA_PATH, "intermediates", "FINAL_RESULTS", "attention_weighted_sentence_embs_np.npy"))
lhs_embs_np = np.load(os.path.join(DATA_PATH, "intermediates", "FINAL_RESULTS", "lhs_embs_np.npy"))
layer_pooled_embs_np = np.load(os.path.join(DATA_PATH, "intermediates", "FINAL_RESULTS", "layer_pooled_embs_np.npy"))
labels_np = np.load(os.path.join(DATA_PATH, "intermediates", "FINAL_RESULTS", "labels_np.npy"))
decoded_reports = np.load(os.path.join(DATA_PATH, "intermediates", "FINAL_RESULTS", "decoded_reports.npy"), allow_pickle=True)

## Plotting params ##
# matplotlib rcparams turn off border
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
# matplotlib rcparams set font
plt.rcParams['font.family'] = 'Arial'
# matplotlib rcparams remove legend border
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = 'black'

# Title style
title_font = {
    'family': 'Arial',
    'color':  'black',
    'weight': 'bold',
    'size': 12,
}
# Master color key
main_col = "#C73682" #"#9F2B68" # Purple
ASD_col = "#23a2dc" # "#FF8719" # Blue
nonASD_col = "#DC5D23" # "#1991ff" # Orange

#### PREDICTING TRUE DIAGNOSIS ####


n_layers = layer_pooled_embs_np.shape[1]

# Make color palette from #F091F3 in seaborn
pal = sns.color_palette("blend:#d3d3d3,"+main_col, n_colors=n_layers)
plt.figure(figsize=(8,5))

with pal:

    ax = plt.subplot()
    layer_auc_list = []
    for i in range(1, n_layers):
        print(f"==== Layer {i} ====")
        pooled_embs = layer_pooled_embs_np[:,i,:]
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=22)

        layer_avg = []
        fold_probas = []
        y_true = []
        auc_list = []
        for train_index, test_index in skf.split(pooled_embs, label_tensor.numpy(), groups=pat_ids):
            X_train, X_test = pooled_embs[train_index], pooled_embs[test_index]
            y_train, y_test = label_tensor.numpy()[train_index], label_tensor.numpy()[test_index]
            clf = LogisticRegression(random_state=22, max_iter=1000).fit(X_train, y_train)
            layer_avg.append(clf.score(X_test, y_test))
            fold_probas.extend(list(clf.predict_proba(X_test)[:,1]))
            y_true.extend(y_test)
            auc_list.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

        fpr, tpr, thresholds = roc_curve(y_true, fold_probas)
        ax.plot(fpr, tpr, label=f"Layer {i}")
        ax.set_xlabel("False Positive Rate", fontdict={"size": 14})
        ax.set_ylabel("True Positive Rate", fontdict={"size": 14})
        # ax.set_title("ROC Curve for Layer-wise Pooled Embeddings, Predicting Diagnosis")
        ax.legend()
        # Calculate AUC for each layer
        print(f"AUC: {np.trapz(tpr, fpr)}")


        print(f"Average accuracy: {np.mean(layer_avg)}")
        print(f"Average AUC: {np.mean(auc_list)}")
        # Print 95% confidence interval for AUC
        print(f"95% CI: {np.percentile(auc_list, [2.5, 97.5])}")
        # Print standard deviation of AUC
        print(f"Standard deviation: {np.std(auc_list)}")
        layer_auc_list.append(auc_list)

# Save figure at high resolution
# Dashed line for AUC of 0.5
ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.6)

plt.savefig(os.path.join(ROOT, "final_plots", "ROC_layer_arial.svg"), dpi=300, bbox_inches="tight")

# Plot AUC per layer (line plot) using layer_auc_list, with 95% CI (error bars) #
layer_auc_list = np.array(layer_auc_list)

# Calculate mean and 95% CI for each layer
mean_auc = np.mean(layer_auc_list, axis=1)
lower_ci = np.percentile(layer_auc_list, 2.5, axis=1)
upper_ci = np.percentile(layer_auc_list, 97.5, axis=1)

# Plot
plt.figure(figsize=(8,5))
# Make each point a different color from the palette
for i in range(1, n_layers):
    plt.errorbar(i, mean_auc[i-1], fmt='o', yerr=[[mean_auc[i-1]-lower_ci[i-1]], [upper_ci[i-1]-mean_auc[i-1]]], color=pal[i-1], capsize=5)
plt.xlabel("Layer", fontsize=14)
plt.ylabel("AUC", fontsize=14)
# Connect dots with lines
plt.plot(range(1, n_layers), mean_auc, linestyle='dashed', color="black", linewidth=0.8)
# Start y-axis at 0.5
plt.ylim(0.7, 1.0)
# X axis interval = 1
plt.xticks(range(1, n_layers));
plt.savefig(os.path.join(ROOT, "final_plots", "AUC_layer_arial.svg"), dpi=300, bbox_inches="tight")

# Set figure size
plt.figure(figsize=(10, 7))
# attn_weighted_sentences = attention_weighted_sentence_embs_np.reshape(-1, 768) # attention weighted
sentence_embs = lhs_embs_np.mean(axis=2).reshape(-1, 768) # LHS
targs = np.array([[item]*64 for item in labels_np]).reshape(-1)
logs = np.array([[item]*64 for item in logits_np]).reshape(-1)
pats = np.array([[item]*64 for item in pd.get_dummies(pat_ids).values.argmax(1)]).reshape(-1)
pca = PCA(2)
pca_embs = pca.fit_transform(sentence_embs)
legend_map = {0: "Non-autism", 1: "Autism"}
sc_targets = sns.scatterplot(x=pca_embs[:, 0], y=pca_embs[:, 1], s=2, hue=np.vectorize(legend_map.get)(targs.astype(int)), palette=[nonASD_col, ASD_col], rasterized=True)
plt.legend(title='MD-Assessed Diagnosis', loc='upper right', prop={"size": 12}, title_fontproperties={"size": 12, "weight": "bold"})
plt.xlabel("PC1", fontdict={"size": 14})
plt.ylabel("PC2", fontdict={"size": 14})
plt.savefig(os.path.join(ROOT, "final_plots", "pca_embs_arial.svg"), dpi=300, bbox_inches="tight")


att_pal = sns.color_palette("blend:#f9f9f9,"+ASD_col, as_cmap=True)

### Look at most attended sentences ###

ind = 3337 # Example report
attns = attention_matrices_np[ind]
ax = sns.heatmap(attention_matrices_np[ind], cmap=att_pal, vmax=1, vmin=0, cbar_kws={'label': 'Attention Weight'}) # magma_r
# plt.title("Attention Heatmap")
plt.ylabel("Sentence Index")
plt.xlabel("Sentence Index")

# Define the positions of the ticks
x_ticks_positions = np.arange(9, attention_matrices_np[ind].shape[1], 10)
x_ticks_positions = np.hstack((0, x_ticks_positions))
y_ticks_positions = np.arange(9, attention_matrices_np[ind].shape[0], 10)
y_ticks_positions = np.hstack((0, y_ticks_positions))

ax.xaxis.set_label_position('top')

# Set the ticks
plt.xticks(x_ticks_positions, x_ticks_positions+1)
plt.yticks(y_ticks_positions, y_ticks_positions+1, rotation=0)

# Put x axis on top
plt.gca().xaxis.tick_top()

max_att = attns.sum(axis=0).argmax()
second_max_att = attns.sum(axis=0).argsort()[::-1][1]
print(f"1. {decoded_reports[ind][max_att]}")
print(f"2. {decoded_reports[ind][second_max_att]}")
print(f"True: {labels_np[ind]}")
print(f"Pred: {np_sigmoid(logits_np[ind]).round()} ({np_sigmoid(logits_np[ind])})")
# Print the report ID
print(f"Report ID: {report_id_array[ind]}")

# Set font sizes
plt.xlabel("Sentence Index", fontsize=12)
plt.ylabel("Sentence Index", fontsize=12)
# Increase colorbar size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.ax.yaxis.label.set_size(14)

# Put rectangle over the most attended column
ax.add_patch(Rectangle((max_att-0.5, 0.15), 2, 63.65, fill=False, edgecolor="black", lw=1.5, alpha=1))

# Save figure at high resolution
plt.savefig(os.path.join(ROOT, "final_plots", f"att_matrix_{ind}_arial.png"), dpi=300, bbox_inches="tight")

# Compare most frequently used words in most attended sentences autism vs control

# Get the most attended sentences
most_attended_sentences_asd = []
most_attended_sentences_ctl = []
n_asd = 0
n_ctl = 0
for i, attn in enumerate(attention_matrices_np):
    if labels_np[i] == 1:
        most_attended_sentences_ind = attn.sum(axis=0).argmax()
        most_attended_sentences_asd.append(decoded_reports[i][most_attended_sentences_ind])
        n_asd += 1
    else:
        most_attended_sentences_ind = attn.sum(axis=0).argmax()
        most_attended_sentences_ctl.append(decoded_reports[i][most_attended_sentences_ind])
        n_ctl += 1

from collections import Counter
# Get the most frequent words in autism
asd_freq_words = []
for sent in most_attended_sentences_asd:
    asd_freq_words.extend([word.strip().strip(punctuation) for word in sent.split()])
asd_freq_words = Counter(asd_freq_words)
for item, count in asd_freq_words.items():
    asd_freq_words[item] = count / n_asd

# Get the most frequent words in control
ctl_freq_words = []
for sent in most_attended_sentences_ctl:
    ctl_freq_words.extend([word.strip().strip(punctuation) for word in sent.split()])
ctl_freq_words = Counter(ctl_freq_words)
for item, count in ctl_freq_words.items():
    ctl_freq_words[item] = count / n_ctl

# Find words with highest difference in frequency
asd_freq_words = pd.DataFrame.from_dict(asd_freq_words, orient="index", columns=["asd_freq"])
ctl_freq_words = pd.DataFrame.from_dict(ctl_freq_words, orient="index", columns=["ctl_freq"])
freq_words = pd.concat([asd_freq_words, ctl_freq_words], axis=1)
freq_words["diff"] = freq_words["asd_freq"] / freq_words["ctl_freq"]
freq_words = freq_words.sort_values(by="diff", ascending=False)

# Plot frequency differences of top words
plt.figure(figsize=(10,3))
sns.barplot(y=freq_words.index, x=freq_words["diff"], orient="h", color=main_col)
plt.xlabel("Frequency Differential (Autism/Non-autism)", fontsize=12)
plt.ylabel("Word", fontsize=12)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%dx'))
# Save figure at high resolution
plt.savefig(os.path.join(ROOT, "final_plots", "word_freq_arial.svg"), dpi=300, bbox_inches="tight")

### Load DSM-5 Criteria and run through model ###
report_max_length = 64
sentence_max_length = 64

DSM_PATH = os.path.join(DATA_PATH, "DSM_5", "txt")
dsm_files = os.listdir(DSM_PATH)

criteria_dict = {"A": [], "B": []}
for file in dsm_files:
    if file.endswith(".txt"):
        with open(os.path.join(DSM_PATH, file), "r") as f:
            lines = [line.rstrip('\n') for line in f]
            criteria_dict[file.split(".")[0]] = lines

tokenizer = AutoTokenizer.from_pretrained(
    "flaubert/flaubert_base_cased", 
    model_max_length = sentence_max_length,
    do_lowercase = False
)

crit_input_ids_dict = {}
crit_att_masks_dict = {}
for key in criteria_dict.keys():
    # Tokenize each criteria
    tokenizer_output = tokenizer(criteria_dict[key], truncation = True, padding = "max_length")
    tokenized_crit = tokenizer_output["input_ids"]
    crit_att_mask = tokenizer_output["attention_mask"]

    # Pad the criteria to the max sentence length
    num_sentences_crit = len(tokenized_crit)
    padding = np.full((report_max_length-num_sentences_crit, sentence_max_length), fill_value=2)
    attention_padding = np.full((report_max_length-num_sentences_crit, sentence_max_length), fill_value=0)

    padded_input_ids = np.vstack((tokenized_crit, padding))
    padded_attention_mask = np.vstack((crit_att_mask, attention_padding))

    crit_input_ids_dict[key] = torch.tensor(padded_input_ids)
    crit_att_masks_dict[key] = torch.tensor(padded_attention_mask)

device = "cpu"
model = model.to(device)
crit_outputs_dict = {}
for key in crit_input_ids_dict.keys():
    input_ids_crit = crit_input_ids_dict[key].to(device)
    att_masks_crit = crit_att_masks_dict[key].to(device)

    crit_logit, crit_att_mat, crit_att_weight_emb, _, lhs_emb, _ = model(input_ids_crit.unsqueeze(0), attn_mask=att_masks_crit.unsqueeze(0))
    crit_outputs_dict[key] = {
        "logit": crit_logit.detach().cpu().numpy(),
        "att_mat": crit_att_mat.squeeze().detach().cpu().numpy(),
        "att_weight_emb": crit_att_weight_emb.squeeze().detach().cpu().numpy(),
        "lhs_emb": lhs_emb.squeeze().detach().cpu().numpy()
    }

### Compute the cosine similarity between every sentence (final layer embeddings, before final sentence attention) and each criteria ###

cos_sim_dict = {"A": [], "B": []}
for i in range(len(lhs_embs_np)):
    sents = lhs_embs_np[i].squeeze().mean(axis=1)
    # A
    A_cos_sim = []
    for j in range(4):
        A_cos_sim_j = F.cosine_similarity(
            torch.tensor(sents),
            torch.tensor(crit_outputs_dict["A"]["lhs_emb"].mean(axis=1)[j]),
            dim=1
        )   
        A_cos_sim.append(A_cos_sim_j.numpy())
    cos_sim_dict["A"].append(A_cos_sim)
    
    # B
    B_cos_sim = []
    for k in range(5):
        B_cos_sim_k = F.cosine_similarity(
            torch.tensor(sents),
            torch.tensor(crit_outputs_dict["B"]["lhs_emb"].mean(axis=1)[k]),
            dim=1
        )
        B_cos_sim.append(B_cos_sim_k.numpy())
    cos_sim_dict["B"].append(B_cos_sim)

cos_sim_dict["A"] = np.stack(cos_sim_dict["A"])[:, 1:, :]
cos_sim_dict["B"] = np.stack(cos_sim_dict["B"])[:, 1:, :]
    
# put A into a dataframe
sim_A = cos_sim_dict["A"]
most_attended_sentences_A = []
for j in range(sim_A.shape[1]):
    sim_A_attended_sents = sim_A[:,j,:][np.arange(sim_A.shape[0]), np.argmax(attention_matrices_np.mean(axis=1), axis=1)]
    most_attended_sentences_A.append(sim_A_attended_sents)
most_attended_sentences_A = np.stack(most_attended_sentences_A).T
most_attended_sentences_A = pd.DataFrame(most_attended_sentences_A, columns=["A1", "A2", "A3"])
most_attended_sentences_A["diagnosis"] = labels_np
# Melting the dataframe
most_attended_sentences_A = pd.melt(most_attended_sentences_A, id_vars=["diagnosis"], value_vars=["A1", "A2", "A3"], var_name="criteria", value_name="cos_sim")

# put B into a dataframe
sim_B = cos_sim_dict["B"]
most_attended_sentences_B = []
for k in range(sim_B.shape[1]):
    sim_B_attended_sents = sim_B[:,k,:][np.arange(sim_B.shape[0]), np.argmax(attention_matrices_np.mean(axis=1), axis=1)]
    most_attended_sentences_B.append(sim_B_attended_sents)
most_attended_sentences_B = np.stack(most_attended_sentences_B).T
most_attended_sentences_B = pd.DataFrame(most_attended_sentences_B, columns=["B1", "B2", "B3", "B4"])
most_attended_sentences_B["diagnosis"] = labels_np
# Melting the dataframe
most_attended_sentences_B = pd.melt(most_attended_sentences_B, id_vars=["diagnosis"], value_vars=["B1", "B2", "B3", "B4"], var_name="criteria", value_name="cos_sim")

all_most_attended_sentences = pd.concat([most_attended_sentences_A, most_attended_sentences_B])


palette = [nonASD_col, ASD_col]
sigma = 0.05
width = 1
ort = "h"
alpha = 0.75
move = 0
f, ax = plt.subplots(figsize=(12, 6))
ax=pt.RainCloud(x="criteria", y="cos_sim", data=all_most_attended_sentences, hue="diagnosis", palette=palette, bw=sigma,
                 width_viol=width, ax=ax, orient="v", alpha=alpha, dodge=True, move=move, box_showfliers=False, box_medianprops={"zorder": 11}, rain_alpha=0, offset=0.2,
                 scale = "area", width_box=0.3, rasterized=True)
plt.xlabel("DSM-5 Criterion", fontdict={"size": 12})
plt.ylabel("Cosine Similarity to Most Attended Sentence", fontdict={"size": 12})
handles, labels = ax.get_legend_handles_labels()
ax.legend(title="MD-Assessed Diagnosis", handles=handles, labels=["Non-autism", "Autism"], loc="lower right", frameon=True, ncol=1, fontsize=12)
# Set legend title font size and weight
plt.setp(ax.get_legend().get_title(), fontsize='12', weight='bold')
ax.autoscale(True)
ax.use_sticky_edges = False
ax.margins(0.02)
# horizontal line at 0
plt.axhline(0, color='lightgrey', linewidth=1, linestyle="--")
plt.show()
# Save figure at high resolution
f.savefig(os.path.join(ROOT, "final_plots", "cosine_dists_arial.png"), dpi=600, bbox_inches="tight")

### Use cosine similarities of most attended sentences for each criteria to predict diagnosis ###
sim_A_attended_sents = sim_A[np.arange(sim_A.shape[0]), :, np.argmax(attention_matrices_np.mean(axis=1), axis=1)]
sim_B_attended_sents = sim_B[np.arange(sim_B.shape[0]), :, np.argmax(attention_matrices_np.mean(axis=1), axis=1)]


# Now do random sentence, not most attended (control)
sim_A_attended_sents_random = sim_A[np.arange(sim_A.shape[0]), :, np.random.randint(0, attention_matrices_np.shape[1], size=attention_matrices_np.shape[0])]
sim_B_attended_sents_random = sim_B[np.arange(sim_B.shape[0]), :, np.random.randint(0, attention_matrices_np.shape[1], size=attention_matrices_np.shape[0])]

sim_combined = np.concatenate((sim_A_attended_sents, sim_B_attended_sents), axis=1)
sim_combined_random = np.concatenate((sim_A_attended_sents_random, sim_B_attended_sents_random), axis=1)
y_numpy = label_tensor.numpy()

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=22)

results_dict = {}

for sim_combined in [sim_combined, sim_combined_random]:
    coefs = []
    avg_score = []
    fold_probas = []
    y_true = []
    for train_index, test_index in skf.split(sim_combined, y_numpy, groups=pat_ids):
        X_train, X_test = sim_combined[train_index], sim_combined[test_index]
        y_train, y_test = y_numpy[train_index], y_numpy[test_index]

        clf = LinearDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)

        scores = clf.transform(X_test).squeeze()

        scores_corrs_fold = []
        for i in range(X_train.shape[1]):
            scores_corrs_fold.append(np.corrcoef(X_test[:,i], scores)[0,1])
        coefs.append(scores_corrs_fold)
            
        avg_score.append(clf.score(X_test, y_test))
        fold_probas.extend(clf.predict_proba(X_test)[:,1])
        y_true.extend(y_test)
        print(clf.score(X_test, y_test))

    fpr, tpr, thresholds = roc_curve(y_true, fold_probas)

    if sim_combined is sim_combined_random:
        results_dict["sim_combined_random"] = [fpr, tpr, coefs]
    else:
        results_dict["sim_combined"] = [fpr, tpr, coefs]


    print(f"Average accuracy: {np.mean(avg_score)}")
    print(f"Standard deviation: {np.std(avg_score)}")

    # Boxplot of coefficients
    coefs = np.stack(coefs)
    box = plt.figure()
    sns.boxplot(data=coefs, showfliers=False, color=main_col, boxprops=dict(linewidth=0), whiskerprops=dict(color=main_col), capprops=dict(color=main_col))
    plt.ylabel("Correlation with LDA Score")
    plt.xlabel("DSM-5 Criteria")
    criteria_names = ["A1", "A2", "A3", "B1", "B2", "B3", "B4"]
    if len(criteria_names) == coefs.shape[1]:
        plt.xticks(np.arange(0, coefs.shape[1]), criteria_names)
    plt.title("5-Fold CV, Predicting Diagnosis")
    plt.suptitle("Box Plot of LDA Scores Correlated with Input Features for DSM-5 Criteria")
    # hline at 0
    plt.axhline(y=0, color='k', linestyle='--')
    plt.show()

    # ROC curve
    roc = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Box Plot of LDA Scores Correlated with Input Features for DSM-5 Criteria")
    # Add legend
    if sim_combined is sim_combined_random:
        plt.legend(["Most Attended Sentence", "Random Sentence"])
    plt.show()


# ROC curve for most attended sentences vs random sentences
roc = plt.figure(figsize=(8,5))
plt.plot(results_dict["sim_combined"][0], results_dict["sim_combined"][1], color=main_col)
plt.plot(results_dict["sim_combined_random"][0], results_dict["sim_combined_random"][1], color="#699296")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
# Add legend
plt.legend(["Most Attended Sentence", "Random Sentence"], loc="lower right", prop={"size": 12})
# Dashed line for AUC of 0.5
plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.6)
# Save figure at high resolution
plt.savefig(os.path.join(ROOT, "final_plots", "ROC_cos_sim_arial.svg"), dpi=300, bbox_inches="tight")

# Box plot of most attended sentences
plt.figure(figsize=(8, 5))
coefs = np.stack(results_dict["sim_combined"][2])
sns.swarmplot(data=coefs, color=main_col, size=7, alpha=1)
# sns.boxplot(data=coefs, showfliers=False, color=main_col, boxprops=dict(linewidth=0), whiskerprops=dict(color=main_col), capprops=dict(color=main_col), width=0.4)
plt.ylabel("Correlation with LDA Score", fontsize=12)
plt.xlabel("DSM-5 Criteria", fontsize=12)
criteria_names = ["A1", "A2", "A3", "B1", "B2", "B3", "B4"]
if len(criteria_names) == coefs.shape[1]:
    plt.xticks(np.arange(0, coefs.shape[1]), criteria_names)
# hline at 0
plt.axhline(y=0, color='k', linestyle='--')
# Save figure at high resolution
plt.savefig(os.path.join(ROOT, "final_plots", "LDA_cos_sim_bp_arial.svg"), dpi=300, bbox_inches="tight")

# Plot the DSM criteria embeddings in the same space as the sentence embeddings
plt.figure(figsize=(6, 5.2))
legend_map = {0: "Non-autism", 1: "Autism"}
sc_targets = sns.scatterplot(x=pca_embs[:, 0], y=pca_embs[:, 1], s=2, hue=np.vectorize(legend_map.get)(targs.astype(int)), palette=[nonASD_col, ASD_col], rasterized=True)
plt.legend(title='MD-Assessed Diagnosis', loc='upper right', prop={"size": 12}, title_fontproperties={"size": 12, "weight": "bold"})
# plt.title('Semantic PCA Embedding Space, with DSM-5 Criteria Labeled');
plt.xlabel("PC1", fontdict={"size": 14})
plt.ylabel("PC2", fontdict={"size": 14})

from matplotlib.patheffects import withStroke

# Now plot the DSM criteria embeddings on the same plot
# Get the embeddings for each criteria
A_embs = crit_outputs_dict["A"]["lhs_emb"].mean(axis=1)[1:4]
B_embs = crit_outputs_dict["B"]["lhs_emb"].mean(axis=1)[1:5]
# Concatenate them
crit_embs = np.concatenate((A_embs, B_embs), axis=0)
# PCA
pca_crit_embs = pca.transform(crit_embs)
# Plot
sc_crit_embs = plt.scatter(x=pca_crit_embs[:, 0], y=pca_crit_embs[:, 1], s=20, c=main_col, rasterized=True)
sc_crit_embs.set_path_effects([withStroke(linewidth=2, foreground='white')])

# use adjustText to avoid overlap
texts = []
for i, txt in enumerate(["A1", "A2", "A3", "B1", "B2", "B3", "B4"]):
    text = plt.text(pca_crit_embs[i, 0], pca_crit_embs[i, 1], txt, fontdict=dict(color=main_col, weight="bold", size=14))
    text.set_path_effects([withStroke(linewidth=1, foreground='white')])

    texts.append(text)
adjust_text(texts)
plt.savefig(os.path.join(ROOT, "final_plots", "pca_crit_embs_arial.svg"), dpi=300, bbox_inches="tight")

    
