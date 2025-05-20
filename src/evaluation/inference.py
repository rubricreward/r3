import json
import os
import argparse
import logging
from functools import partial
import concurrent.futures
import threading
import time

from datasets import load_dataset

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(CUR_DIR)))
CLIENT_RETRIES = 3
DEBUG_COUNT = 10
MODEL = None
SAMPLING_PARAMS = None
TOKENIZER = None

EVAL_DATASETS = [
    "rubricreward/R3-eval-FeedbackBench",
    "rubricreward/R3-eval-reward-bench"
    "rubricreward/R3-eval-XSUM",
    "rubricreward/R3-eval-MMLU-STEM",
    "rubricreward/R3-eval-RM-Bench",
    "rubricreward/R3-eval-BBH",
]

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def request_openai_completion(input_item, model_config):
    # Set OpenAI client
    from openai import OpenAI
    
    model_id = model_config.get('model_id')
    if "deepseek" in model_id:
        base_url = "https://api.deepseek.com"
    else:
        base_url = "https://api.openai.com/v1/"
    openai_client = OpenAI(base_url=base_url)

    for attempt in range(CLIENT_RETRIES):
        try:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=input_item['msg'],
                **model_config['generation_args']
            )

            if model_id == "deepseek-reasoner":
                result = {
                    "id": input_item["id"],
                    "reasoning_content": response.choices[0].message.reasoning_content,
                    "response": response.choices[0].message.content
                }
            else:
                result = {
                    "id": input_item["id"],
                    "reasoning_content": "No reasoning",
                    "response": response.choices[0].message.content
                }
            
            return result
        except Exception as e:
            logging.warning("Error calling OpenAI API:")
            time.sleep(61)

    logging.warning(f"Could not resolve error after {CLIENT_RETRIES} attempts for input ID: {input_item['id']}")
    return {
        "id": input_item["id"],
        "reasoning_content": "Error",
        "response": "Error"
    }

def openai_completion(messages, model_config, output_file_name, num_workers=8):    
    lock = threading.Lock()  # for thread-safe file writing
    partial_func = partial(request_openai_completion, model_config=model_config)

    with open(output_file_name, 'a', encoding='utf-8') as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_input = {
                executor.submit(partial_func, input_): input_
                for input_ in messages
            }

            for future in concurrent.futures.as_completed(future_to_input):
                res = future.result()
                if res is not None:
                    with lock:  # ensure safe file writing
                        f.write(json.dumps(res, ensure_ascii=False) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                        
def vllm_completion(messages, dataset_ids, output_file_name):
    list_text = TOKENIZER.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
    
    outputs = MODEL.generate(list_text, SAMPLING_PARAMS)

    # Print the outputs.
    result_list = []
    for output, id_ in zip(outputs, dataset_ids):
        result_list.append({
            'id': id_,
            'response': output.outputs[0].text
        })
        
    # Read existing data from the file (if it exists)
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    with open(output_file_name, 'w', encoding='utf-8') as f:
        data.extend(result_list)
        json.dump(data, f, indent=4)
                        
def get_dataset(dataset_name, split='train'):
    dataset = load_dataset(dataset_name)[split]
    
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

def generate_response(dataset_name, split, output_file_name,
                      model_config, debug):
    # Get dataset
    dataset_ids, messages = get_dataset(dataset_name, split)
    
    if debug:
        dataset_ids = dataset_ids[:DEBUG_COUNT]
        messages = messages[:DEBUG_COUNT]
    
    logging.info(f"Length dataset of {dataset_name}: {len(messages)}")
    
    # Perform inference
    use_vllm = model_config.get('use_vllm')
    if use_vllm:
        vllm_completion(messages, dataset_ids, output_file_name)
    else:
        openai_completion(messages, model_config, output_file_name)

# prepare the model input
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on R3 Evaluation Dataset(s)')
    parser.add_argument('--model_config_path', '-c', type=str, required=True,
                        help=f"Model's config for running evaluation. For example, see `data/configs`.")
    parser.add_argument('--dataset_names', '-d', type=str, default="all",
                        help="List of dataset to be evaluated upon, separated by comma(s). `all` means infer on all.")
    parser.add_argument('--output_folder', '-o', type=str, default='output',
                        help="Output folder name.")
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help=f"Debug with {DEBUG_COUNT} samples.")
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    config_path = args.model_config_path.strip()
    model_config = {}
    if os.path.exists(config_path):
        config_abs_path = os.path.join(ROOT_DIR, config_path)
        if not os.path.exists(config_path):
            config_abs_path = os.path.join(ROOT_DIR, config_path)
            if not os.path.exists(config_abs_path):
                raise ValueError(f"Config path `{config_path}` does not exist!")
            else:
                logging.warning(f"Config path `{config_path}` is not found, switching to `{config_abs_path}`")
                config_path = config_abs_path
    
    if not config_abs_path.endswith('.json'):
        raise NotImplementedError("Config path is not in JSON Format, other format is not implemented yet!")
    else:
        with open(config_abs_path, 'r') as f:
            model_config = json.load(f)
            
    dataset_names = args.dataset_names.strip()
    eval_dataset_list = []
    if dataset_names == "all":
        eval_dataset_list = EVAL_DATASETS
    else:
        dataset_name_list = dataset_names.split(",")
        for dataset_name in dataset_name_list:
            if dataset_name in EVAL_DATASETS:
                eval_dataset_list.append(dataset_name)
            else:
                logging.warning(f"Unrecognized evaluation dataset named `{dataset_name}`, skipping ...")

    if len(eval_dataset_list) == 0:
        raise ValueError("Evaluation datasets cannot be empty!")

    # Create output folder
    output_folder = args.output_folder
    if not os.path.isabs(output_folder):
        output_folder = os.path.join(ROOT_DIR, args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Check if need to initialize vLLM
    use_vllm = model_config.get('use_vllm', False)
    model_id = model_config.get('model_id')
    if use_vllm:
        # Initialize vLLM model globally (once)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        import torch
        
        model_id = model_config.get('model_id')
        tensor_parallel_size = model_config.get('tensor_parallel_size', torch.cuda.device_count())
        TOKENIZER = AutoTokenizer.from_pretrained(model_id)
        SAMPLING_PARAMS = SamplingParams(**model_config.get("generation_args", {}))
        
        MODEL = LLM(model_id, tensor_parallel_size=tensor_parallel_size, **model_config.get("model_args", {}))

    for dataset_name in eval_dataset_list:
        save_name = f"{'-'.join(dataset_name.split('-')[2:])}.json"
        save_path = os.path.join(output_folder, save_name)
        if dataset_name != "rubricreward/R3-eval-XSUM":
            generate_response(dataset_name, "train", save_path,
                              model_config, args.debug)
        else:
            for metric in ["faithfulness", "coherence", "relevance"]:
                generate_response(dataset_name, metric, save_path,
                                   model_config, args.debug)
