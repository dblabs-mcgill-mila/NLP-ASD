import re
import math
import numpy as np
import pandas as pd
from cleantext import clean

import torch
from transformers import AutoTokenizer


def split_string_with_min_length(input_string, min_length=30):
    """
    Split an input_string into sentences with a min_length; if
    a sentence is shorter than min_length, it is added to the
    previous sentence.

    Args:
        input_string (str): String to split
        min_length (int): Minimum length of a sentence
    Returns:
        result (list): List of sentences
    """
    # Split string at periods
    splits = input_string.split('.')
    # Remove empty string from list
    splits = [split for split in splits if split]

    result = []
    for split in splits:
        # If split is longer than min_length, add it to result
        if len(split) > min_length or len(result) == 0:
            string = split + '.'
            result.append(string.strip())
        # If split is shorter than min_length, add it to last item in result
        else:
            result[-1] += split + '.'

    return result

def process_clean_data(report_files, meta_frame, sentence_min_length=30):
    """
    Takes a list of .txt report_files (list of full path to files)
    and a meta_frame of meta data and returns a dict with
    each report split into sentences, the diagnosis, the report ID,
    and the patient ID.
    The text data are cleaned.

    Args:
        report_files (list): list of full path to report .txt files
        meta_frame (DataFrame): meta data frame
        sentence_min_length (int): minimum length of sentences to not be attached
                                    to the previous sentence
    Returns:
        data_df (DataFrame): DataFrame with each report split into sentences,
                                the diagnosis, the report ID, and the patient ID

    """
    data_dict = {}
    for file in report_files:
        report_id = file.split('/')[-1].split('.')[0]
        if report_id in meta_frame.code.values:
            final_diag = meta_frame[meta_frame.code == report_id].final_diag.values[0]
            if not math.isnan(final_diag):
                final_diag = int(final_diag)
                report = open(file, "r").read()
                # Default cleaning
                clean_report = clean(
                    report,
                    to_ascii = False,
                    lower = True,
                    no_line_breaks = True,
                    lang = "fr",
                    no_punct = False
                )
                # Remove multiple periods, underscores, hyphens
                to_remove = "_.-"
                pattern = "(?P<char>[" + re.escape(to_remove) + "])(?P=char)+"
                clean_report = re.sub(pattern, r"\1", clean_report)
                # Remove all asterisks
                clean_report = clean_report.replace("*", "")

                # Split string into sentences
                sentences = split_string_with_min_length(clean_report, min_length=sentence_min_length)

                final_diag = int(meta_frame[meta_frame.code == report_id].final_diag.values[0])
                pat_id = report_id[:7]
                count = 0
                for sentence in sentences:
                    data_dict[report_id + f"_{count}"] = [sentence, final_diag, report_id, pat_id]
                    count += 1

    
    
    data_df = pd.DataFrame(data_dict).T
    data_df.columns = ["text", "labels", "report_id", "pat_id"]

    return data_df 



def hierarchical_tokenizer(
    report_files, 
    meta_frame, 
    tokenizer_name,
    sentence_max_length,
    sentence_min_length,
    report_max_length
):
    """
    This function will take a list of paths to .txt files and return a tensor 
    of tokenized sentences grouped by report. Both the report and sentence level 
    will be padded to make each level symmetrical. Will also return a tensor of
    of attention masks for the input_ids tensor, as well as a tensor of labels
    and a numpy array of report ids.

    Args:
        report_files (list): List of paths to .txt files
        meta_frame (pandas.DataFrame): DataFrame containing report_id and labels
        tokenizer_name (str): Name of HuggingFace tokenizer to use
        sentence_max_length (int): Maximum number of tokens per sentence
        sentence_min_length (int): Minimum number of tokens per sentence
        report_max_length (int): Maximum number of sentences per report
    
    Returns:
        all_input_ids (torch.tensor): Tensor of tokenized sentences (N x S x T)
        all_attention_mask (torch.tensor): Tensor of attention masks (N x S x T)
        all_diagnosis (torch.tensor): Tensor of labels (N x 1)
        all_report_ids (numpy.array): Array of report ids (N x 1)
    """

    # Clean and process reports
    data_df = process_clean_data(report_files, meta_frame, sentence_min_length)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
        model_max_length = sentence_max_length,
        do_lowercase = False
    )

    # Tokenize sentences
    data_df['tokenized_sentences'] = data_df['text'].apply(
        lambda x: tokenizer(x, truncation = True, padding = "max_length")
    )

    # Collect tokenized sentences per report
    all_input_ids = []
    all_attention_mask = []
    all_diagnosis = []
    all_report_ids = []
    for report_id in data_df.report_id.unique():
        report_tokenized = data_df[data_df.report_id == report_id].tokenized_sentences
        report_diagnosis = data_df[data_df.report_id == report_id].labels[0]
        num_sentences_report = len(report_tokenized)

        # Get the input ids and attention masks for each sentence, and stack them
        report_input_ids = np.stack(report_tokenized.map(lambda v: v["input_ids"]))
        report_attention_mask = np.stack(report_tokenized.map(lambda b: b["attention_mask"]))

        # If the number of sentences in the report is greater than the max length,
        # truncate the report to the max length. Otherwise, pad the report to the
        # max length (and extend the attention mask to ignore paddings).
        if num_sentences_report > report_max_length:
            padded_report_input_ids = report_input_ids[:report_max_length]
            padded_report_attention_mask = report_attention_mask[:report_max_length]
        else:
            padding = np.full((report_max_length-num_sentences_report, sentence_max_length), fill_value=2)
            attention_padding = np.full((report_max_length-num_sentences_report, sentence_max_length), fill_value=0)
            padded_report_input_ids = np.vstack((report_input_ids, padding))
            padded_report_attention_mask = np.vstack((report_attention_mask, attention_padding))
        
        all_input_ids.append(padded_report_input_ids)
        all_attention_mask.append(padded_report_attention_mask)
        all_diagnosis.append(report_diagnosis)
        all_report_ids.append(report_id)
    
    all_input_ids = torch.tensor(np.stack(all_input_ids))
    all_attention_mask = torch.tensor(np.stack(all_attention_mask))
    all_diagnosis = torch.tensor(all_diagnosis)
    all_report_ids = np.array(all_report_ids)

    return all_input_ids, all_attention_mask, all_diagnosis, all_report_ids
    