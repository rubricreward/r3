from .rubric_prompt_gen import generate_rubric, generate_prompt_df, ROOT_DIR, RUBRICS_DIR

import os
import pandas as pd
from datasets import Dataset, DatasetDict
import json

XSUM_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'others', 'xsum_scores_annotated.csv')
XSUM_RUBRIC_PATH = {
    'faithfulness': os.path.join(RUBRICS_DIR, "xsum_faithfulness_rubric.json"),
    'coherence': os.path.join(RUBRICS_DIR, "xsum_coherence_rubric.json"),
    'relevance': os.path.join(RUBRICS_DIR, "xsum_relevance_rubric.json"),
}

METRICS_LIST = ["coherence", "relevance"]

def generate_prompt_xsum():
    df = pd.read_csv(XSUM_CSV_PATH)
    dataset = Dataset.from_pandas(df)
    metric_datasets = {}

    for metric in METRICS_LIST:
        new_df = generate_prompt_df(df=dataset, rubric_load_path=XSUM_RUBRIC_PATH[metric],
                                evaluation_mode='pointwise', input_data_column='article', response_column='summary')
        if metric == "coherence":
            new_df = new_df.remove_columns(["faithfulness", "relevance"])
        elif metric == "relevance":
            new_df = new_df.remove_columns(["faithfulness", "coherence"])
        new_df = new_df.rename_column(metric, "score")
        new_df = new_df.add_column("id", list(range(len(new_df))))
        metric_datasets[metric] = new_df
        
    new_df = generate_prompt_df(df=dataset, rubric_load_path=XSUM_RUBRIC_PATH['faithfulness'],
                                evaluation_mode='binary', input_data_column='article', response_column='summary')
    new_df = new_df.remove_columns(METRICS_LIST)
    new_df = new_df.rename_column("faithfulness", "score")
    new_df = new_df.add_column("id", list(range(len(new_df))))
    metric_datasets["faithfulness"] = new_df

    metric_datasets = DatasetDict(metric_datasets)

    return metric_datasets
