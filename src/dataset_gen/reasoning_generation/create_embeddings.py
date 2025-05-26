import re
import random
import argparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

RANDOM_SEED = 42
EMBED_INSTRUCT = "Represent the following evaluation task and input based on the reasoning and topic it covers:\n\n{prompt}"

def get_embed_instruction(prompt):
    task_input = re.search(r'### TASK\s+(.*?)\s+### RESPONSE', prompt, re.DOTALL)
    return EMBED_INSTRUCT.format(prompt=task_input)

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 1. Embed documents in batches
def embed_input_prompts(input_prompt_list, model_id="Alibaba-NLP/gte-Qwen2-7B-instruct", batch_size=32):
    formatted_instr_prompt = [get_embed_instruction(prompt) for prompt in input_prompt_list]
    model = SentenceTransformer(model_id, trust_remote_code=True)
    prompt_embeddings = model.encode(formatted_instr_prompt, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return prompt_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embed prompt column of the dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help=f"Dataset id path.")
    parser.add_argument('--embedding_path', '-d', type=str,
                        help="Embedding path.")
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    # Load dataset from huggingface, then embed the task along with the instruction
    set_seed()
    dataset = load_dataset(args.dataset_path)
    emb = embed_input_prompts(dataset['prompt'])
    np.save(args.embedding_path, emb)
    