import os
import logging
from functools import partial
import concurrent.futures
import threading
import json
import time
import argparse

from openai import OpenAI
from transformers import AutoTokenizer

from datasets import load_dataset

OPENAI_RETRIES = 3
OPENAI_CLIENT = OpenAI()
MAX_REASON_TOKENS = 4096

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def request_openai_completion(input_item):
    for attempt in range(OPENAI_RETRIES):
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4.1-mini",
                messages=input_item['msg'],
                max_tokens=MAX_REASON_TOKENS,
            )

            result = {
                "id": input_item["id"],
                "response": response.choices[0].message.content
            }
            
            return result
        except Exception as e:
            logging.warning("Error calling OpenAI API:")
            time.sleep(61)
    logging.warning(f"Could not resolve error after {OPENAI_RETRIES} attempts for input ID: {input_item['id']}")
    return {
        "id": input_item["id"],
        "response": "Error"
    }

def openai_completion(batched_input, output_path, num_workers=8):
    lock = threading.Lock()  # for thread-safe file writing

    with open(output_path, 'a', encoding='utf-8') as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_input = {
                executor.submit(partial(request_openai_completion, input)): input
                for input in batched_input
            }

            for future in concurrent.futures.as_completed(future_to_input):
                res = future.result()
                if res is not None:
                    with lock:  # ensure safe file writing
                        f.write(json.dumps(res, ensure_ascii=False) + '\n')
                        f.flush()
                        os.fsync(f.fileno())

def generate_responses(batched_dataset, output_path):
    def format_data(data):
        instruction = "Shorten the following reasoning trace by removing redundant or unnecessary thinking loops while preserving the exact same tone, style, and progression of thought. Output only the shortened reasoning trace without any explanation."
        return {
            "msg": [{"role": "user", "content": f"{instruction}\n\n{data['r1_reasoning_trace']}"}],
        }

    batched_input = batched_dataset.map(format_data).to_list()
    openai_completion(batched_input, output_path)

def update_dataset_with_summarization(dataset, id_to_reason):
    def add_new_trace(example):
        if example['id'] in id_to_reason:
            example['r1_reasoning_trace_summarized'] = id_to_reason[example['id']]
        else:
            example['r1_reasoning_trace_summarized'] = example['r1_reasoning_trace']
        return example

    def add_raw_trace_summarized(example):
        example['r1_raw_sft_summarized'] = f"<think>{example['r1_reasoning_trace_summarized']}</think>{example['r1_response']}"
        return example
    
    updated_dataset = dataset.map(add_new_trace)
    updated_dataset = updated_dataset.map(add_raw_trace_summarized)
    return updated_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize reasoning from DeepSeek-R1')
    parser.add_argument('--output_path', type=str, required=True,
                        help="Reasoning summarization output path.")
    parser.add_argument('--max_reason_tokens', type=int, default=MAX_REASON_TOKENS,
                        help="Max reason tokens to be summarized.")
    args = parser.parse_args()
    
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    dataset = load_dataset("rubricreward/R3-Dataset-14K")['train']

    # Collect indices of long examples
    long_indices = []
    for idx, example in enumerate(dataset):
        tokenized = tokenizer(example['r1_raw_sft'], truncation=False, add_special_tokens=False)
        if len(tokenized['input_ids']) > args.max_reason_tokens:
            long_indices.append(example['id'])

    # Summarize reasoning traces
    filtered_dataset = dataset.filter(lambda example: example['id'] in long_indices)
    generate_responses(filtered_dataset, args.output_path)
    
    # Update reasoning traces to summarized version
    id_to_reason = {}
    with open(args.output_path, 'r') as f:
        for line in f.readlines():
            json_line = json.loads(line)
            id_to_reason[json_line['id']] = json_line['response']

    # This will be the new dataset with the summarized reasoning trace
    updated_dataset = update_dataset_with_summarization(filtered_dataset, id_to_reason)
