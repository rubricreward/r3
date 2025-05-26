import os
import pandas as pd
from datasets import load_dataset, Dataset

from .rubric_prompt_gen import generate_prompt_df, RUBRICS_DIR

def generate_skyworks_prompt():
    df = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")['train']
    formatted_data = []
    for row in df:
        if row['chosen'][0]['role'] == 'user':
            input_message = row['chosen'][0]['content']
            chosen_response = row['chosen'][1]['content']
        else:
            input_message = row['chosen'][1]['content']
            chosen_response = row['chosen'][0]['content']
            
        if row['rejected'][0]['role'] == 'user':
            rejected_response = row['rejected'][1]['content']
        else:
            rejected_response = row['rejected'][0]['content']

        # Store the reformatted structure
        formatted_data.append({
            "input": input_message,  # User input
            "chosen": chosen_response,  # Assistant's preferred response
            "rejected": rejected_response,  # Assistant's rejected response
            "source": row["source"]  # Keep source for reference
        })
    
    result_ds = Dataset.from_pandas(pd.DataFrame(formatted_data))

    result_ds = generate_prompt_df(df=result_ds, rubric_load_path=os.path.join(RUBRICS_DIR, "skyworks", "skyworks_rubric.json"),
                                   evaluation_mode='pairwise', input_data_column='input',
                                   response_column='chosen', second_response_column='rejected')
    return result_ds
