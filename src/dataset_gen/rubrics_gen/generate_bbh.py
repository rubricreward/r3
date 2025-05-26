import os
import json
import argparse
from pathlib import Path
from typing import Any, Optional

import re
import numpy as np
import pandas as pd
from outlines import generate, models
from outlines.models.openai import OpenAIConfig
from datasets import load_dataset, Dataset

from .rubric_prompt_gen import (
    EvaluationRubric,
    _answer_parser,
    _generate_examples,
    _get_detailed_instruct,
    _generate_embeddings,
    _retrieve_examples,
    BINARY_RUBRIC_TEMPLATE_PATH,
    BinaryScoreSchema,
    evaluate_binary
)

# The task instructions were based from https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/cot-prompts
bbh_subsets = {
    "boolean_expressions": "Evaluate the result of a random Boolean expression.",
    "causal_judgement": "Answer questions about causal attribution.",
    "date_understanding": "Infer the date from context.",
    "disambiguation_qa": "Clarify the meaning of sentences with ambiguous pronouns.",
    "dyck_languages": "Correctly close a Dyck-n word.",
    "formal_fallacies": "Distinguish deductively valid arguments from formal fallacies.",
    "geometric_shapes": "Name geometric shapes from their SVG paths.",
    "hyperbaton": "Order adjectives correctly in English sentences.",
    "logical_deduction_five_objects": "A logical deduction task which requires deducing the order of a sequence of objects.",
    "logical_deduction_seven_objects": "A logical deduction task which requires deducing the order of a sequence of objects.",
    "logical_deduction_three_objects": "A logical deduction task which requires deducing the order of a sequence of objects.",
    "movie_recommendation": "Recommend movies similar to the given list of movies.",
    "multistep_arithmetic_two": "Solve multi-step arithmetic problems.",
    "navigate": "Given a series of navigation instructions, determine whether one would end up back at the starting point.",
    "object_counting": "Questions that involve enumerating objects and asking the model to count them.",
    "penguins_in_a_table": "Answer questions about a table of penguins and their attributes.",
    "reasoning_about_colored_objects": "Answer extremely simple questions about the colors of objects on a surface.",
    "ruin_names": "Select the humorous edit that 'ruins' the input movie or musical artist name.",
    "salient_translation_error_detection": "Detect the type of error in an English translation of a German source sentence.",
    "snarks": "Determine which of two sentences is sarcastic.",
    "sports_understanding": "Determine whether an artificially constructed sentence relating to sports is plausible or not.",
    "temporal_sequences": " Answer questions about which times certain events could have occurred.",
    "tracking_shuffled_objects_five_objects": "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    "tracking_shuffled_objects_seven_objects": "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    "tracking_shuffled_objects_three_objects": "A task requiring determining the final positions of a set of objects given their initial positions and a description of a sequence of swaps.",
    "web_of_lies": "Evaluate a random boolean function expressed as a word problem.",
    "word_sorting": "Sort a list of words.",
}

def create_prompt(
    task_description: str,
    fewshot: int,
    sample_rubrics: list[dict[str, str]],
) -> str:
    system_prompt = """
        You are an expert evaluator. Given a defined task, analyze the task and create a rubric using a Likert scale from 1 to 5 to that will help to perform the given task. 
        Please follow these steps:
        1. Explain the criteria for distinguishing between the scores (e.g., how a score of 1 differs from a score of 5).
        2. Based on your analysis, generate a rubric in JSON format with the Likert scale ranging from 1 to 5, including descriptions for each score.
        3. Ensure that the rubric is clear, actionable, and covers key aspects of the task.
        """

    def _add_fewshot(fewshot):
        return (
            "Below are some examples:\n"
            + "".join(
                f"Question: {eg.get('input')} Answer: {eg.get('target')}\n"
                for eg in fewshot
            )
            if len(fewshot) > 0
            else ""
        )

    prompt = f"""
        {system_prompt}

        ### Task
        {task_description}
        {_add_fewshot(fewshot)}

        {_generate_examples(sample_rubrics, title="### Example rubrics (Unrelated Tasks)")}


        ### Rubric for current task
        """

    return prompt

def get_sample_rubrics(
    rubrics_path: Path,
    task: str,
    embedding_model: str = "intfloat/multilingual-e5-large-instruct",
):
    with open(rubrics_path, "r") as f:
        list_of_rubrics = json.load(f)

    embed_instruction = "Given a task description, retrieve the proper evaluation metrics to solve such task."
    instruct_rubrics = [
        _get_detailed_instruct(embed_instruction, rubric.get("Description"))
        for rubric in list_of_rubrics
    ]
    instruct_task = _get_detailed_instruct(embed_instruction, task)

    query_embedding = _generate_embeddings([instruct_task], embedding_model)[0]
    rubric_embeddings = _generate_embeddings(instruct_rubrics, embedding_model)
    sample_rubrics = _retrieve_examples(
        query_embedding, rubric_embeddings, list_of_rubrics
    )
    return sample_rubrics

def generate_rubrics():
    parser = argparse.ArgumentParser("Generate rubric for all BBH tasks.")
    parser.add_argument("--output_path", type=Path, help="Path to store the rubric results in JSONL.")
    parser.add_argument("--n_fewshot", type=int, default=3, help="Number of fewshot examples to show for a given task.")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="OpenAI model to use for rubric generation.")
    parser.add_argument("--sample_rubrics", type=Path, default=None, help="Path to sample rubrics.")
    args = parser.parse_args()

    all_generated_rubrics: dict[str, dict[str, Any]] = {}
    for task_name, task_description in bbh_subsets.items():
        print(f"Generating rubric for {task_name}")

        # Set-up fewshot examples for a given task
        df = load_dataset("lukaemon/bbh", task_name, split="test").to_pandas()
        fewshot = (
            df.sample(args.n_fewshot).to_dict("records") if args.n_fewshot > 0 else []
        )
        sample_rubrics = get_sample_rubrics(args.sample_rubrics, task=task_description)

        # Create a prompt
        prompt = create_prompt(
            task_description,
            fewshot=fewshot,
            sample_rubrics=sample_rubrics,
        )

        # Set-up generator
        openai_config = OpenAIConfig(temperature=0.7, seed=42)
        model = models.openai(args.openai_model, config=openai_config)
        generator = generate.json(model, EvaluationRubric)

        # Generate rubric
        response = generator(prompt)
        all_generated_rubrics[task_name] = _answer_parser(response)

    if args.output_path:
        with open(args.output_path, "w") as outfile:
            json.dump(all_generated_rubrics, outfile, indent=2)
