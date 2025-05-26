import json
import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

from .rubric_prompt_gen import generate_prompt_df, ROOT_DIR, RUBRICS_DIR

NEGATIVE_SAMPLED_NQ_PATH = os.path.join(ROOT_DIR, 'data', 'others', 'negative_added_NQ.json')
NEGATIVE_SAMPLED_TQ_PATH = os.path.join(ROOT_DIR, 'data', 'others', 'negative_added_TQ.json')

def generate_prompt_evouna():
    formatted_data = []

    with open(NEGATIVE_SAMPLED_NQ_PATH, 'r') as f:
        nq_df = json.load(f)

    for row in nq_df:
        # Store the reformatted structure
        formatted_data.append({
            "ID": f"{row['ID']}-true",  # Keep source for reference
            "input": row["question"],  # User input
            "response": row["golden_answer"],  # Assistant's preferred response
            "score": "true"
        })
        
        formatted_data.append({
            "ID": f"{row['ID']}-false",  # Keep source for reference
            "input": row["question"],  # User input
            "response": row["negative_answer"],  # Assistant's preferred response
            "score": "false"
        })
        
    with open(NEGATIVE_SAMPLED_TQ_PATH, 'r') as f:
        tq_df = json.load(f)
        
    for row in tq_df:
        # Store the reformatted structure
        formatted_data.append({
            "ID": f"{row['ID']}-true",  # Keep source for reference
            "input": row["question"],  # User input
            "response": row["golden_answer"],  # Assistant's preferred response
            "score": "true"
        })
        
        formatted_data.append({
            "ID": f"{row['ID']}-false",  # Keep source for reference
            "input": row["question"],  # User input
            "response": row["negative_answer"],  # Assistant's preferred response
            "score": "false"
        })

    result_ds = Dataset.from_pandas(pd.DataFrame(formatted_data))

    result_ds = generate_prompt_df(df=result_ds, rubric_load_path=os.path.join(RUBRICS_DIR, 'evouna', 'evouna_rubric.json'),
                                evaluation_mode='binary', input_data_column='input', response_column='response')
    return result_ds
