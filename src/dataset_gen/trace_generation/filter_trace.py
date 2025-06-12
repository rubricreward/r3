import json
import os
import argparse
import re
import logging 

import pandas as pd
from datasets import load_dataset

TOKENIZER = None
MODEL = None
SAMPLING_PARAMS = None

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def extract_score(text):
    match = re.search(
        r'(?:\\?"score\\?"|score)\s*:\s*'
        r'(?:"(Response 1|Response 2|true|false|True|False|\d+(?:\.\d+)?)"|'
        r'(Response 1|Response 2|true|false|True|False|\d+(?:\.\d+)?))',
        text
    )
    if match:
        for group in match.groups():
            if group is not None:
                # Convert numeric strings to float or int
                if re.match(r'^\d+\.\d+$', group):
                    return int(float(group))
                elif re.match(r'^\d+$', group):
                    return int(group)
                return group  # Return as-is
    return None

def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name)['train']
    
    def format_data(data):
        return {
            "msg": [{"role": "user", "content": data.get("prompt")}],
        }

    dataset_ids = []
    messages = []
    batched_input = dataset.map(format_data).to_list()
    for inp in batched_input:
        dataset_ids.append(inp["id"])
        messages.append(inp["msg"])
    
    return dataset_ids, messages

def generate_responses(dataset_name, sampling_params, file_name):
    dataset_ids, messages = get_dataset(dataset_name)
    
    logging.info(f"Length dataset: {len(messages)}")

    list_text = TOKENIZER.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False
                                            )
    
    outputs = MODEL.generate(list_text, sampling_params)

    # Print the outputs.
    result_list = []
    for output, id_ in zip(outputs, dataset_ids):
        result_list.append({
            'id': id_,
            'response': output.outputs[0].text
        })
        
    # Read existing data from the file (if it exists)
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    with open(file_name, 'w', encoding='utf-8') as f:
        data.extend(result_list)
        json.dump(data, f, indent=4)

def extract_r1_score(example):
    example['r1_score'] = extract_score(example['r1_response'])
    return example

# prepare the model input
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter examples in multiple steps')
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Output folder path")
    parser.add_argument('--filter_stage', type=int, default=1, choices=[1, 2],
                        help="Filter stage")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    
    # Assume there is column r1_response and actual_score; this gives `rubricreward/R3-Dataset-14K`
    dataset = load_dataset("rubricreward/R3-Dataset-20K", split='train')
    dataset = dataset.map(extract_r1_score)
    filtered_dataset = dataset.filter(lambda example: str(example['actual_score']) == str(example['r1_score']))
    
    # This gives `rubricreward/R3-Dataset-4K`
    if args.filter_stage == 2:
        # Initialize the tokenizer
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        MODEL = LLM(model="Qwen/Qwen3-4B", max_model_len=10000, enforce_eager=True)

        # Generate outputs
        for seed in range(5):
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=1024,
                                             min_p=0, top_k=20, seed=seed)
            generate_responses(filtered_dataset, sampling_params, os.path.join(args.output_folder, f"eval_qwen3_4b_{seed}.json"))
            
        # Load generated responses across seeds
        records = []
        for seed in range(5):
            file_path = os.path.join(args.output_folder, f"eval_qwen3_4b_{seed}.json")
            with open(file_path, 'r') as f:
                data = json.load(f)
            for entry in data:
                records.append({
                    'id': entry['id'],
                    'seed': seed,
                    'response_score': extract_score(entry['response'])
                })
        
        df_responses = pd.DataFrame(records)
        filtered_dataset_df = filtered_dataset.to_pandas()

        # Merge with actual scores
        df_merged = pd.merge(df_responses, filtered_dataset_df, on='id', how='left')

        # Check correctness per response
        df_merged['response_score'] = df_merged['response_score'].astype(str)
        df_merged['actual_score'] = df_merged['actual_score'].astype(str)
        df_merged['num_correct_qwen3_4b'] = df_merged['response_score'] == df_merged['actual_score']

        # Count correct per ID
        summary = df_merged.groupby('id')['num_correct_qwen3_4b'].sum().reset_index()

        # Filter IDs where correct_count != 5
        ids_to_keep = set(summary[summary['num_correct_qwen3_4b'] == 5]['id'])
        filtered_dataset = filtered_dataset.filter(lambda example: example['id'] in ids_to_keep)
