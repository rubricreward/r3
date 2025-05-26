import time
import json
import logging
import argparse
import concurrent.futures

import outlines
import openai
from openai import OpenAI
from tqdm import tqdm

OPENAI_RETRIES = 3

OPENAI_CLIENT = OpenAI()

MODEL_LIST = ['fid', 'gpt35', 'chatgpt', 'gpt4', 'newbing']

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

@outlines.prompt
def construct_wrong_answer(question, correct_answer, wrong_answer_list=None):
    """
    You are given a question and its correct answer.
    If applicable, you will also be given a list of incorrect answers.
    Your task is to:
    - Extract a single plausible but incorrect answer from the provided false answers.
    - If none of the given false answers are suitable, generate a new one that is factually incorrect but sounds reasonable.
    
    ### Examples
    
    #### Example 1
    Question: which mode is used for short wave broadcast service
    Correct Answer: Olivia/MFSK
    Wrong Answers:
    - Shortwave broadcast service is typically used in the AM broadcasting mode.
    - The mode used for short wave broadcast service is amplitude modulation (AM).
    - Hello, this is Bing.  According to , shortwave radio is a type of radio transmission technology that uses shortwave frequencies (between 3. 3 and 30 MHz) to carry voice or music.  There are two different types of shortwave transmissions: AM and FM.  For most types of voice communication (with the exception of international broadcast stations), you will need a radio with sideband mode.
    
    Response: AM
    
    #### Example 2
    Question: what does hp mean in war and order
    Correct Answer: hit points or health points
    
    Response: Hewlett-Packard
    
    #### Example 3
    Question: Who was the man behind The Chipmunks?
    Correct Answer: David Seville
    Wrong Answers:
    - The Chipmunks were created by Ross Bagdasarian Sr. in 1958.
    
    Response: Ross Bagdasarian Sr.
    
    ### Task
    Question: {{ question }}
    Correct Answer: {{ correct_answer }}
    {% if wrong_answer_list %}
    Wrong Answers:
    {% for wr in wrong_answer_list %}
    - {{ wr }}
    {% endfor %}
    
    {% endif %}
    Response:
    """
    
@outlines.prompt
def correct_wrong_answer(question, correct_answer, wrong_answer):
    """
    You are given a question, its correct answer, and an extracted negative answer.  
    Your task is to return only the incorrect answer as a single phrase or term, without explanations or extra text.  

    ### Examples
    
    #### Example 1
    Question: which mode is used for short wave broadcast service
    Correct Answer: Olivia/MFSK
    Negative Answer: AM
    Response: AM
    
    #### Example 2
    Question: who wrote the book the origin of species
    Correct Answer: Charles Darwin
    Negative Answer: Alfred Russel Wallace  \n\n(Note: Wallace was a contemporary of Darwin who independently conceived the idea of natural selection, making this a plausible but incorrect answer.)
    Response: Alfred Russel Wallace
    
    #### Example 3
    Question: who died in the plane crash greys anatomy
    Correct Answer: Lexie Grey
    Negative Answer: One plausible but incorrect answer could be:  \n\n**Dr. Derek Shepherd (McDreamy)**  \n\n(While Derek Shepherd did die in a later season, it
    Response: Derek Shepherd
    
    ### Task
    Question: {{ question }}
    Correct Answer: {{ correct_answer }}
    Negative Answer:: {{ wrong_answer }}
    Response:
    """

def add_prompt_fix_answer(input_item):
    golden_answer = input_item['golden_answer']
    wrong_answer = input_item['negative_answer']
    return [{"role": "user",
             "content": correct_wrong_answer(input_item.get("question"), golden_answer, wrong_answer)}]
    
def add_prompt_construct_answer(input_item):
    wrong_responses = []
    for model in MODEL_LIST:
        if not input_item.get(f"judge_{model}"):
            wrong_responses.append(input_item.get(f"answer_{model}"))

    if len(wrong_responses) == 0:
        wrong_responses = None
    golden_answer = input_item['golden_answer']
            
    return [{"role": "user",
             "content": construct_wrong_answer(input_item.get("question"), golden_answer, wrong_responses)}]

def openai_completion(batched_input):
    def process_input(input_item):
        for attempt in range(OPENAI_RETRIES):
            try:
                response = OPENAI_CLIENT.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=input_item['msg'],
                    max_tokens=32,
                    temperature=0.01
                )
                result = input_item
                result.pop('msg', None)
                result['negative_answer'] = response.choices[0].message.content
                return result
            except openai.OpenAIError as e:
                if "rate" in str(e).lower():
                    logging.warning("Hit rate limit; retrying...")
                    time.sleep(61)
                else:
                    logging.exception("Error calling OpenAI API:")
                    raise e
        logging.exception(f"Could not resolve error after {OPENAI_RETRIES} attempts for input ID: {input_item['data_id']}")
        return None

    # Using ThreadPoolExecutor to process batched_input concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_input = {executor.submit(process_input, inp): inp for inp in batched_input}
        for future in tqdm(concurrent.futures.as_completed(future_to_input), total=len(future_to_input), desc="OpenAI requests"):
            res = future.result()
            if res is not None:
                results.append(res)
                
    return results

def add_negative_answers(old_json_path, new_json_path):
    with open(old_json_path, 'r') as f:
        cur_inputs = json.load(f)
        
    for i in range(len(cur_inputs)):
        cur_inputs[i]['msg'] = add_prompt_construct_answer(cur_inputs[i])
    
    results = openai_completion(cur_inputs)
    with open(new_json_path, 'w') as f:
        json.dump(results, f, indent=4)

def fix_negative_answers(old_json_path, new_json_path):
    with open(old_json_path, 'r') as f:
        cur_inputs = json.load(f)
        
    for i in range(len(cur_inputs)):
        cur_inputs[i]['msg'] = add_prompt_fix_answer(cur_inputs[i])
    
    results = openai_completion(cur_inputs)
    with open(new_json_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform original EVOUNA dataset with proper negative answer.')
    parser.add_argument('--original_evouna_path', type=str, required=True,
                        help="Original EVOUNA dataset path.")
    parser.add_argument('--modified_evouna_path', type=str, required=True,
                        help="Modified EVOUNA dataset path.")
    args = parser.parse_args()

    # Step 1, extract negative answer from EVOUNA's collection of responses
    add_negative_answers(args.original_evouna_path, args.modified_evouna_path)
    
    # Step 2 (Optional), fix (potentially wrong) negative answer by performing answer extraction    
    fix_negative_answers(args.modified_evouna_path, args.modified_evouna_path)
