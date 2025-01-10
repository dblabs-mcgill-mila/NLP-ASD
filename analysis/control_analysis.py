### Imports ###

import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from custom_models.sentence_attention_base_pool_hf import SentenceAttentionBERT

import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns

from adjustText import adjust_text

### Load Data ###

# Local path to files
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

input_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "input_tensor"))
attention_mask_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "attention_mask_tensor"))
label_tensor = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "label_tensor"))
report_id_array = torch.load(os.path.join(DATA_PATH, "reports_tokenized", "report_id_array"))

pat_ids = []
for report_id in report_id_array:
    pat_ids.append(report_id[:7])

meta_frame = pd.read_csv(
    os.path.join(DATA_PATH, "metadata.csv")
)

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

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def compute_metrics(p):
    pred, labels = p
    # pred = np.argmax(pred, axis=1)
    pred = np.where(sigmoid(pred) > 0.5, 1, 0)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Make a new column of patient age groups, where 0 is below median, 1 is above median
median_age = meta_frame['pat_age'].median()
meta_frame['age_group_med'] = 0
meta_frame.loc[meta_frame['pat_age'] >= median_age, 'age_group_med'] = 1
meta_frame.index = meta_frame['code']

med_age_labels = torch.tensor(meta_frame.loc[report_id_array]['age_group_med'].values)

label_dict = {}
label_dict["med_age"] = med_age_labels
label_dict[f"shuffle_0"] = label_tensor[torch.randperm(len(label_tensor))]

report_max_length = 64
sentence_max_length = 64
DSM_PATH = os.path.join(DATA_PATH, "DSM_5")
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

all_most_attended_sentences_list = []
for label_type in label_dict.keys():
    data_dict = {
        "input_ids": input_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": label_dict[label_type],
    }
    dataset = Dataset.from_dict(data_dict)

    hf_dir = os.path.join(os.environ['SCRATCH'], "hugging_face")

    training_args = TrainingArguments(
        output_dir=hf_dir,
        num_train_epochs=2,
        warmup_steps=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
        fp16=True,
        optim="paged_adamw_32bit",
        logging_dir="./logs",
        report_to="none"
    )

    model = SentenceAttentionBERT("flaubert/flaubert_base_cased", report_max_length=64, ff_dropout=0.15, att_dropout=0, class_dropout=0, sentence_dropout=0) # Use best HP tune config
    model = model.to("cuda")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    outputs = []
    atts = []
    labels_np = [] 
    for batch in trainer.get_train_dataloader():
        with torch.no_grad():
            lhs, att = model(**batch, inference=True)
            outputs.append(lhs.mean(axis=2).detach().cpu().numpy())
            atts.append(att.detach().cpu().numpy())
        labels_np.append(batch['labels'].detach().cpu().numpy())
    outputs = np.vstack(outputs)
    atts = np.vstack(atts).squeeze()
    labels_np = np.vstack(labels_np)
    labels_np = np.hstack(labels_np)

    crit_outputs_dict = {}
    for key in crit_input_ids_dict.keys():
        input_ids_crit = crit_input_ids_dict[key].to("cuda")
        att_masks_crit = crit_att_masks_dict[key].to("cuda")
        fake_labels = torch.zeros(1).type(torch.LongTensor).to("cuda")

        with torch.no_grad():
            crit_lhs_emb, _ = model(input_ids_crit.unsqueeze(0), att_masks_crit.unsqueeze(0), fake_labels, inference=True)
            crit_outputs_dict[key] = {
                "lhs_emb": crit_lhs_emb.squeeze().detach().cpu().numpy()
            }
    
    cos_sim_dict = {"A": [], "B": []}
    for i in range(len(outputs)):
        sents = outputs[i].squeeze()
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
        sim_A_attended_sents = sim_A[:,j,:][np.arange(sim_A.shape[0]), np.argmax(atts.mean(axis=1), axis=1)]
        most_attended_sentences_A.append(sim_A_attended_sents)
    most_attended_sentences_A = np.stack(most_attended_sentences_A).T
    most_attended_sentences_A = pd.DataFrame(most_attended_sentences_A, columns=["A1", "A2", "A3"])
    # most_attended_sentences_A["diagnosis"] = labels_np
    most_attended_sentences_A["diagnosis"] = label_tensor.numpy()
    # Melting the dataframe
    most_attended_sentences_A = pd.melt(most_attended_sentences_A, id_vars=["diagnosis"], value_vars=["A1", "A2", "A3"], var_name="criteria", value_name="cos_sim")
    # put B into a dataframe
    sim_B = cos_sim_dict["B"]
    most_attended_sentences_B = []
    for k in range(sim_B.shape[1]):
        sim_B_attended_sents = sim_B[:,k,:][np.arange(sim_B.shape[0]), np.argmax(atts.mean(axis=1), axis=1)]
        most_attended_sentences_B.append(sim_B_attended_sents)
    most_attended_sentences_B = np.stack(most_attended_sentences_B).T
    most_attended_sentences_B = pd.DataFrame(most_attended_sentences_B, columns=["B1", "B2", "B3", "B4"])
    # most_attended_sentences_B["diagnosis"] = labels_np
    most_attended_sentences_B["diagnosis"] = label_tensor.numpy()
    # Melting the dataframe
    most_attended_sentences_B = pd.melt(most_attended_sentences_B, id_vars=["diagnosis"], value_vars=["B1", "B2", "B3", "B4"], var_name="criteria", value_name="cos_sim")

    all_most_attended_sentences = pd.concat([most_attended_sentences_A, most_attended_sentences_B])

    all_most_attended_sentences_list.append(all_most_attended_sentences)

    palette = [nonASD_col, ASD_col]
    sigma = 0.05
    width = 1
    ort = "h"
    alpha = 0.75
    move = 0

    f, ax = plt.subplots(figsize=(12, 6))
    ax=pt.RainCloud(x="criteria", y="cos_sim", data=pd.concat(all_most_attended_sentences_list), hue="diagnosis", palette=palette, bw=sigma,
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
    plt.ylim(-0.5, 1)
    # horizontal line at 0
    plt.axhline(0, color='lightgrey', linewidth=1, linestyle="--")
    plt.show()
    # Save figure at high resolution
    f.savefig(os.path.join(ROOT, "final_plots", "cosine_dists_shuffle_asd_label_arial2.png"), dpi=600, bbox_inches="tight")

    ### Use cosine similarities of most attended sentences for each criteria to predict diagnosis ###
    sim_A_attended_sents = sim_A[np.arange(sim_A.shape[0]), :, np.argmax(atts.mean(axis=1), axis=1)]
    sim_B_attended_sents = sim_B[np.arange(sim_B.shape[0]), :, np.argmax(atts.mean(axis=1), axis=1)]


    # # Do top n sentences
    # n_top = 2
    # sim_A_attended_sents = sim_A[np.arange(sim_A.shape[0])[:, None], :, np.argpartition(attention_matrices_np.mean(axis=1), kth=-n_top, axis=1)[:,-n_top:]].reshape(sim_A.shape[0], -1)
    # sim_B_attended_sents = sim_B[np.arange(sim_B.shape[0])[:, None], :, np.argpartition(attention_matrices_np.mean(axis=1), kth=-n_top, axis=1)[:,-n_top:]].reshape(sim_A.shape[0], -1)

    # Do random sentence, not most attended (should have lower accuracy)
    sim_A_attended_sents_random = sim_A[np.arange(sim_A.shape[0]), :, np.random.randint(0, atts.shape[1], size=atts.shape[0])]
    sim_B_attended_sents_random = sim_B[np.arange(sim_B.shape[0]), :, np.random.randint(0, atts.shape[1], size=atts.shape[0])]

    sim_combined = np.concatenate((sim_A_attended_sents, sim_B_attended_sents), axis=1)
    sim_combined_random = np.concatenate((sim_A_attended_sents_random, sim_B_attended_sents_random), axis=1)
    y_numpy = label_tensor.numpy()
    # y_numpy = []
    # for label_type in list(label_dict.keys())[2:]:
    #     y_numpy.append(label_dict[label_type].numpy())
    # y_numpy = np.stack(y_numpy).T

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    results_dict = {}

    for sim_combined in [sim_combined, sim_combined_random]:
        coefs = []
        avg_score = []
        fold_probas = []
        y_true = []
        for train_index, test_index in skf.split(sim_combined, y_numpy, groups=pat_ids):
            X_train, X_test = sim_combined[train_index], sim_combined[test_index]
            y_train, y_test = y_numpy[train_index], y_numpy[test_index]
            # Normalize Z-score
            # X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
            # X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

            # clf = LogisticRegression(random_state=50, penalty=None, max_iter=1000).fit(X_train, y_train)
            clf = LinearDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)

            scores = clf.transform(X_test).squeeze()

            scores_corrs_fold = []
            for i in range(X_train.shape[1]):
                scores_corrs_fold.append(np.corrcoef(X_test[:,i], scores)[0,1])
            coefs.append(scores_corrs_fold)

            # coefs.append(clf.coef_.squeeze())

                
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
    roc = plt.figure(figsize=(6,5))
    plt.plot(results_dict["sim_combined"][0], results_dict["sim_combined"][1], color=main_col)
    plt.plot(results_dict["sim_combined_random"][0], results_dict["sim_combined_random"][1], color="#699296")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    # Add legend
    plt.legend(["Most Attended Sentence", "Random Sentence"], loc="lower right", prop={"size": 12})
    # Dashed line for AUC of 0.5
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.6)
    # Save figure at high resolution
    plt.savefig(os.path.join(ROOT, "final_plots", "ROC_cos_sim_med_age_asd_label_arial2.svg"), dpi=300, bbox_inches="tight")

    # swarm plot of most attended sentences
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
    plt.savefig(os.path.join(ROOT, "final_plots", "LDA_cos_sim_bp_shuffle_arial2.svg"), dpi=300, bbox_inches="tight")

    # Set figure size
    plt.figure(figsize=(9, 7))
    sentence_embs = outputs.reshape(-1, 768) # LHS
    targs = np.array([[item]*64 for item in labels_np]).reshape(-1)
    pats = np.array([[item]*64 for item in pd.get_dummies(pat_ids).values.argmax(1)]).reshape(-1)
    pca = PCA(2)
    pca_embs = pca.fit_transform(sentence_embs)
    legend_map = {0: "Label 1", 1: "Label 2"}
    age_pal = ["#DF4EBC", "#4EDF71"]
    rand_pal = ["#474FE2", "#E2474F"]
    sc_targets = sns.scatterplot(x=pca_embs[:, 0], y=pca_embs[:, 1], s=2, hue=np.vectorize(legend_map.get)(targs.astype(int)), palette=rand_pal, rasterized=True)
    plt.legend(title='Random Label', loc='upper right', prop={"size": 12}, title_fontproperties={"size": 12, "weight": "bold"})
    plt.xlabel("PC1", fontdict={"size": 14})
    plt.ylabel("PC2", fontdict={"size": 14})
    plt.savefig(os.path.join(ROOT, "final_plots", "pca_embs_shuffle_arial.svg"), dpi=300, bbox_inches="tight")
    plt.show()

        





