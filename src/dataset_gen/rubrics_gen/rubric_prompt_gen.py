import outlines
from outlines import generate, models
from outlines.models.openai import OpenAIConfig
from pydantic import BaseModel, ConfigDict, Field, constr
from sentence_transformers import SentenceTransformer
import numpy as np
import datasets

from typing import Dict, List, Any
import os
import json
import argparse

from .evaluation_schema import EvaluationRubric, PointwiseScoreSchema, PairwiseScoreSchema, BinaryScoreSchema
from .prompts import evaluate_pointwise, evaluate_pairwise, evaluate_binary

EVALUATION_MODE_DICT = {
    "pointwise": evaluate_pointwise,
    "pairwise": evaluate_pairwise,
    "binary": evaluate_binary
}

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(CUR_DIR)))
RUBRICS_DIR = os.path.join(ROOT_DIR, 'data', 'rubrics')

FEEDBACK_COLLECTION_RUBRIC_PATH = os.path.join(RUBRICS_DIR, "modified_feedback_collection.json")
PAIRWISE_RUBRIC_TEMPLATE_PATH = os.path.join(RUBRICS_DIR, "pairwise_rubric_template.json")
BINARY_RUBRIC_TEMPLATE_PATH = os.path.join(RUBRICS_DIR, "binary_rubric_template.json")

TEMPERATURE = 0.7

# Function to generate embeddings
def _generate_embeddings(questions, retriever_checkpoint):
    model = SentenceTransformer(retriever_checkpoint)
    return model.encode(questions, convert_to_tensor=True, device="cuda", show_progress_bar=True)

# Function to retrieve examples based on embeddings
def _retrieve_examples(query_embedding, train_embeddings, train_records, top_k=5)-> List:
    """
    Retrieves the top_k examples from the train dataset based on cosine similarity.
    """
    if top_k == 0:
        return []

    cosine_similarities = np.dot(train_embeddings.cpu().numpy(), query_embedding.cpu().numpy()) / (
        np.linalg.norm(train_embeddings.cpu().numpy(), axis=1) * np.linalg.norm(query_embedding.cpu().numpy())
    )
    top_indices = np.argsort(-cosine_similarities)[:top_k]
    return [train_records[i] for i in top_indices]

def _get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruction: {task_description}\nQuery: {query}'

def _generate_examples(samples, title="Examples") -> str:
    """
    Generate formatted text for examples or rubrics.
    
    Args:
        samples (list): List of dictionaries containing example data or rubrics.
        title (str): Title for the section (e.g., "Examples" or "Rubrics").
    
    Returns:
        str: Formatted text with examples/rubrics.
    """
    text = [f"{title}"]  # Add title
    for i, samp in enumerate(samples):
        example_text = f"Example {i + 1}:\n{json.dumps(samp, indent=2)}"
        text.append(example_text)
    return "\n".join(text)  # Join with newlines

def _answer_parser(response) -> Dict[str, Any]:
    json_response = {
        'description': response.description,
        'scoring': {f"score_{str(i)}": getattr(response.scoring, f"score_{str(i)}") for i in range(1, 6)}
    }
    return json_response

def generate_pointwise_rubric(generator, task, sample_rubrics) -> Dict[str, Any]:
    system_prompt = """
    You are an expert evaluator. Given a defined task, analyze the task and create a rubric using a Likert scale from 1 to 5 to that will help to perform the given task. 
    Please follow these steps:
    1. Explain the criteria for distinguishing between the scores (e.g., how a score of 1 differs from a score of 5).
    2. Based on your analysis, generate a rubric in JSON format with the Likert scale ranging from 1 to 5, including descriptions for each score.
    3. Ensure that the rubric is clear, actionable, and covers key aspects of the task.
    """

    # Define the ICL prompt
    prompt = f"""
    {system_prompt}
    
    ### Task
    {task}

    {_generate_examples(sample_rubrics, title="### Example rubrics (Unrelated Tasks)")}

    ### Rubric for current task:
    """

    response = generator(prompt)
    return _answer_parser(response)

def generate_rubric(task_desc: str, evaluation_mode: str, rubric_save_path: str,
                    embedding_model="intfloat/multilingual-e5-large-instruct",
                    openai_model="gpt-4o-mini"):
    """Generates rubric depending on evaluation mode and saves it into `rubric_save_path`

    Args:
        task_desc (str): Task description
        evaluation_mode (str): Either pointwise, pairwise, or binary
        rubric_save_path (str): Rubric save path (for inspection or further running)
        embedding_model (str, optional): Embedding model for pointwise rubric generation. Defaults to "intfloat/multilingual-e5-large-instruct".
        openai_model (str, optional): OpenAI model for pointwise rubric generation. Defaults to "gpt-4o-mini".
    """
    if evaluation_mode not in EVALUATION_MODE_DICT:
        raise ValueError(f"Evaluation mode `{evaluation_mode}` not in {list(EVALUATION_MODE_DICT.values())}")
    elif evaluation_mode == "pointwise":
        new_rubric_list = []
        for seed in range(5):
            openai_config = OpenAIConfig(
                temperature=TEMPERATURE,
                seed=seed
            )
            model = models.openai(openai_model, config=openai_config)
            generator = generate.json(model, EvaluationRubric)
            
            # Find most relevant rubric
            with open(FEEDBACK_COLLECTION_RUBRIC_PATH, 'r') as f:
                list_of_rubrics = json.load(f)
                
            embed_instruction = 'Given a task description, retrieve the similar task descriptions.'
            instruct_rubrics = [rubric.get("description") for rubric in list_of_rubrics]
            instruct_task = _get_detailed_instruct(embed_instruction, task_desc)

            query_embedding = _generate_embeddings([instruct_task], embedding_model)[0]
            rubric_embeddings = _generate_embeddings(instruct_rubrics, embedding_model)
            sample_rubrics = _retrieve_examples(query_embedding, rubric_embeddings, list_of_rubrics)

            # Generate representative sample_rubrics
            rubric = generate_pointwise_rubric(generator, task_desc, sample_rubrics)
            new_rubric_list.append(rubric)
            
        with open(rubric_save_path, 'w') as f:
            json.dump(new_rubric_list, f, indent=4)
            
    elif evaluation_mode == "pairwise" or evaluation_mode == "binary":
        rubric_read_path = PAIRWISE_RUBRIC_TEMPLATE_PATH if evaluation_mode == "pairwise" else BINARY_RUBRIC_TEMPLATE_PATH
        new_rubric_list = []
        
        with open(rubric_read_path, 'r') as f:
            rubric_list = json.load(f)
            for cur_rubric in rubric_list:
                new_rubric = {}
                new_rubric['description'] = task_desc
                new_rubric['scoring'] = cur_rubric
                new_rubric_list.append(new_rubric)
                
        with open(rubric_save_path, 'w') as f:
            json.dump(new_rubric_list, f, indent=4)    

def generate_prompt(rubric_load_path: str, evaluation_mode: str, input_data: str, response: str,
                    second_response: str=None, seed: int=0) -> str:
    """
    Args:
        rubric_load_path (str): Path to the rubric
        evaluation_mode (str): Either pointwise, pairwise, or binary
        input_data (str): Input to be evaluated
        response (str): Response from the model to be judged
        second_response (str, optional): Second response to compare for `pairwise`. Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to 0.

    Returns: Prompt generated from model.
    """
    np.random.seed(seed)
    with open(rubric_load_path, 'r') as f:
        rubric_list = json.load(f)

    if evaluation_mode not in EVALUATION_MODE_DICT:
        raise ValueError(f"Evaluation mode `{evaluation_mode}` not in {list(EVALUATION_MODE_DICT.values())}")
    elif evaluation_mode == "pointwise":
        selected_rubric = rubric_list[np.random.randint(0, len(rubric_list))]
        created_rubric = PointwiseScoreSchema(score_1=selected_rubric.get("scoring").get("score_1"),
                                              score_2=selected_rubric.get("scoring").get("score_2"),
                                              score_3=selected_rubric.get("scoring").get("score_3"),
                                              score_4=selected_rubric.get("scoring").get("score_4"),
                                              score_5=selected_rubric.get("scoring").get("score_5"))
        return evaluate_pointwise(selected_rubric.get('description'), input_data, response, created_rubric)
    elif evaluation_mode == "pairwise":
        selected_rubric = rubric_list[np.random.randint(0, len(rubric_list))]
        created_rubric = PairwiseScoreSchema(response_1=selected_rubric.get('response_1'), response_2=selected_rubric.get('response_2'))
        return evaluate_pairwise(selected_rubric.get('description'), input_data, response,
                        second_response, created_rubric)
    elif evaluation_mode == "binary":
        selected_rubric = rubric_list[np.random.randint(0, len(rubric_list))]
        created_rubric = BinaryScoreSchema(true=selected_rubric.get('true'), false=selected_rubric.get('false'))
        return evaluate_binary(selected_rubric.get('description'), input_data, response, created_rubric)

# Function to process a single row
def _process_row(row, index, rubric_list, evaluation_mode, input_data_column, response_column, second_response_column=None) -> Dict[str, Any]:
    # Select a random rubric
    selected_rubric = rubric_list[np.random.randint(0, len(rubric_list))]

    # Extract relevant fields
    input_data = row[input_data_column]
    response = row[response_column]
    second_response = row.get(second_response_column) if second_response_column else None

    if evaluation_mode == "pointwise":
        created_rubric = PointwiseScoreSchema(
            score_1=selected_rubric.get("scoring").get("1"),
            score_2=selected_rubric.get("scoring").get("2"),
            score_3=selected_rubric.get("scoring").get("3"),
            score_4=selected_rubric.get("scoring").get("4"),
            score_5=selected_rubric.get("scoring").get("5")
        )
        return {"prompt": evaluate_pointwise(selected_rubric.get('description'), input_data, response, created_rubric)}

    elif evaluation_mode == "pairwise":
        # For index % 2 != 0, the order is swapped
        if index % 2 != 0 and second_response is not None:
            response, second_response = second_response, response  
        created_rubric = PairwiseScoreSchema(
            response_1=selected_rubric.get("scoring").get("response_1"),
            response_2=selected_rubric.get("scoring").get("response_2")
        )
        return {"prompt": evaluate_pairwise(selected_rubric.get('description'), input_data, response, second_response, created_rubric)}

    elif evaluation_mode == "binary":
        created_rubric = BinaryScoreSchema(
            true=selected_rubric.get("scoring").get('true'),
            false=selected_rubric.get("scoring").get('false')
        )
        return {"prompt": evaluate_binary(selected_rubric.get('description'), input_data, response, created_rubric)}

def generate_prompt_df(df: datasets.Dataset, rubric_load_path: str, evaluation_mode: str,
                       input_data_column: str, response_column: str,
                       second_response_column: str=None, seed: int=0) -> datasets.Dataset:
    """
    Args:
        df (datasets.Dataset): Dataset to be augmented with prompt
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
    
    if evaluation_mode not in EVALUATION_MODE_DICT:
        raise ValueError(f"Evaluation mode `{evaluation_mode}` not in {list(EVALUATION_MODE_DICT.values())}")
    
    with open(rubric_load_path, 'r') as f:
        rubric_list = json.load(f)
    
    new_df = df.map(lambda row, index: _process_row(row, index, rubric_list, evaluation_mode,
                                                  input_data_column, response_column, second_response_column),
                  with_indices=True)
    
    return new_df
