import os
from datasets import load_dataset, Dataset

from .rubric_prompt_gen import generate_prompt_df, RUBRICS_DIR

def generate_rmbench_prompt():
    dataset = load_dataset("THU-KEG/RM-Bench")['train']
    dataset = dataset.rename_column('prompt', 'input')

    final_dataset_list = []

    for example in dataset:
        prompt = example["input"]
        chosen_list = example["chosen"]
        rejected_list = example["rejected"]
        
        for i in range(3):
            for j in range(3):
                chosen = chosen_list[i]
                rejected = rejected_list[j]
                
                final_dataset_list.append({
                    "id": f"{example['id']}-{i}-{j}",
                    "input": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })

    new_ds = Dataset.from_list(final_dataset_list)
    new_ds = generate_prompt_df(df=new_ds, rubric_load_path=os.path.join(RUBRICS_DIR, "rmbench", "rmbench_rubric.json"),
                            evaluation_mode='pairwise', input_data_column='input', response_column='chosen',
                            second_response_column='rejected')

    result_ds = new_ds.map(
        lambda example, idx: {
            'score': 'Response 1' if idx % 2 == 0 else 'Response 2'
        },
        with_indices=True
    )
    
    return result_ds