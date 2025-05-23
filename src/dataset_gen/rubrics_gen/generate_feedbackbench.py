from .prompts import evaluate_pointwise
from .evaluation_schema import PointwiseScoreSchema

import numpy as np
from datasets import load_dataset, Dataset
from typing import Dict, Any

# Function to process a single row
def _process_row(row) -> Dict[str, Any]:
    created_rubric = PointwiseScoreSchema(
        score_1=row.get('orig_score1_description'),
        score_2=row.get('orig_score2_description'),
        score_3=row.get('orig_score3_description'),
        score_4=row.get('orig_score4_description'),
        score_5=row.get('orig_score5_description')
    )
    return {"prompt": evaluate_pointwise(row.get('orig_criteria'), row['orig_instruction'], row['orig_response'], created_rubric)}

def _generate_prompt_ds(ds: Dataset, seed: int=0) -> Dataset:
    """
    Args:
        ds (Dataset): Dataset to be augmented with prompt
        rubric_load_path (str): Path to the rubric
        evaluation_mode (str): Either pointwise, pairwise, or binary
        input_data (str): Input to be evaluated
        response (str): Response from the model to be judged
        second_response (str, optional): Second response to compare for `pairwise`. Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to 0.

    Returns: Prompt generated from model.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    result_ds = ds.map(lambda row: _process_row(row))
    
    return result_ds

def generate_feedback_bench_prompt():
    ds = load_dataset("prometheus-eval/Feedback-Bench")['train']

    result_ds = _generate_prompt_ds(ds)
    result_ds = result_ds.rename_column("orig_score", "score")
    result_ds = result_ds.rename_column("__index_level_0__", "id")
    result_ds = result_ds.remove_columns(["orig_score1_description", "orig_score2_description", "orig_score3_description",
                                          "orig_score4_description", "orig_score5_description",
                                          "input", "instruction", "messages"])

    return result_ds
