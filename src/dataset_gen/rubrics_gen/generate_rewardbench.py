from datasets import load_dataset, Dataset, DatasetDict

from .rubric_prompt_gen import generate_prompt_df, PAIRWISE_RUBRIC_TEMPLATE_PATH

def generate_rewardbench_prompt():
    ds = load_dataset("allenai/reward-bench")['filtered']    
    new_ds = ds.rename_column('prompt', 'input')

    new_ds = generate_prompt_df(df=new_ds, rubric_load_path=PAIRWISE_RUBRIC_TEMPLATE_PATH,
                            evaluation_mode='pairwise', input_data_column='input',
                            response_column='chosen', second_response_column='rejected')

    # Use dataset.map to add a "score" field: Response 1 if index even, else Response 2
    result_ds = new_ds.map(
        lambda example, idx: {
            'score': 'Response 1' if idx % 2 == 0 else 'Response 2'
        },
        with_indices=True
    )

    return result_ds
