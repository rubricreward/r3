import os
import logging
import argparse
from functools import partial
import concurrent.futures
import threading
import json
import time

from openai import OpenAI
from datasets import load_dataset

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
OPENAI_RETRIES = 3
OPENAI_CLIENT = OpenAI(base_url="https://api.deepseek.com")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def request_openai_completion(input_item):
    for attempt in range(OPENAI_RETRIES):
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model="deepseek-reasoner",
                messages=input_item['msg'],
                max_tokens=8192,
            )

            result = {
                "id": input_item["id"],
                "reasoning_content": response.choices[0].message.reasoning_content,
                "response": response.choices[0].message.content
            }
            
            return result
        except Exception as e:
            logging.warning("Error calling OpenAI API:")
            time.sleep(61)
    logging.warning(f"Could not resolve error after {OPENAI_RETRIES} attempts for input ID: {input_item['id']}")
    return {
        "id": input_item["id"],
        "reasoning_content": "Error",
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

def format_data(data):
    return {
        "msg": [{"role": "user", "content": data.get("prompt")}],
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distill reasoning from DeepSeek-R1')
    parser.add_argument('--output_path', type=str, required=True,
                        help="Reasoning distillation output path.")
    args = parser.parse_args()

    dataset = load_dataset("rubricreward/R3-Dataset-20K", split='train')
    batched_input = dataset.map(format_data).to_list()
    openai_completion(batched_input, args.output_path, num_workers=8)
