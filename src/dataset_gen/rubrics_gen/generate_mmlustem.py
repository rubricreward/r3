import random
import os
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .rubric_prompt_gen import generate_prompt, RUBRICS_DIR

OPTION_LABELS = ["(A)", "(B)", "(C)", "(D)"]

def generate_mmlustem_prompt():
    result_df = []
    i = 0

    ds = load_dataset("TIGER-Lab/MMLU-Stem", split="test")
    for example in tqdm(ds, desc=f"Generating dataset"):
        options = example["choices"]
        labeled_options = [f"{OPTION_LABELS[idx]} {opt}" for idx, opt in enumerate(options)]

        input_str = example["question"] + "\nOptions:\n" + "\n".join(labeled_options)

        for answer in ['true', 'false']:
            if answer == 'true':
                answer_idx = example["answer"]
            else:
                incorrect_indices = [idx for idx in range(len(options)) if idx != example["answer"]]
                answer_idx = random.choice(incorrect_indices)

            chosen = OPTION_LABELS[answer_idx]

            prompt = generate_prompt(
                rubric_load_path=os.path.join(RUBRICS_DIR, "mmlustem", "mmlustem_rubric.json"),
                evaluation_mode="binary",
                input_data=input_str,
                response=chosen,
            )

            row = {
                "id": f"{i}-{answer}",
                "prompt": prompt,
                "input": input_str,
                "response": chosen,
                "score": answer,
                "choices": example["choices"],
                "subject": example["subject"],
            }
            result_df.append(row)
        i += 1

    result_ds = Dataset.from_pandas(pd.DataFrame(result_df))
    return result_ds
    