import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import json
import os

from .rubric_prompt_gen import generate_prompt_df, RUBRICS_DIR

METRICS_LIST = ["coherence", "consistency", "fluency", "relevance"]

def generate_summeval_prompt():
    df = load_dataset("davidanugraha/SummEval")["train"]
    result_ds_dict = {}

    for metric in METRICS_LIST:
        def format_data(example):
            return {
                "id": example["id"],
                "input": example["source"],
                "response": example["hyp"],
                "score": example[f"expert_{metric}"]
            }
        
        result_ds = df.map(lambda row: format_data(row)).select_columns(["id", "input", "response", "score"])
        result_ds = generate_prompt_df(df=result_ds, rubric_load_path=os.path.join(RUBRICS_DIR, 'summeval', f'{metric}_rubric.json'),
                                evaluation_mode='pointwise', input_data_column='input', response_column='response')
        result_ds_dict[metric] = result_ds
        
    return result_ds_dict
