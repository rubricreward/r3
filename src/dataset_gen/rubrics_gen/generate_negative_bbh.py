import copy
import os
import json
import random
from pathlib import Path
from functools import partial
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer

from .rubric_prompt_gen import generate_prompt_df, RUBRICS_DIR, ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, 'data')

multiple_choice_tasks = [
    "date_understanding", "disambiguation_qa", "geometric_shapes",
    "hyperbaton",  # technically also binary, but formatted as multi-choice,
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
]
binary_tasks = [
    "boolean_expressions",  # TRUE FALSE
    "causal_judgement",  # Yes / No
    "formal_fallacies",  # valid / invalid
    "navigate",  # Yes / No
    "sports_understanding",  # yes / no
    "web_of_lies",  # Yes / No
]
numerical_tasks = [
    "multistep_arithmetic_two",  # up to 5 digit
    "object_counting",  # 0 - 17

]
custom_tasks = [
    "dyck_languages",  # ] ] >
    "word_sorting",  # word1 word2 word3
]

all_tasks = [*multiple_choice_tasks, *binary_tasks, *numerical_tasks, *custom_tasks]


class BBHTask:

    def __init__(self, task: str):
        self.task = task
        self.dataset = load_dataset("maveriq/bigbenchhard", task)["train"]

    @abstractmethod
    def create_random_negatives(self):
        pass

    @abstractmethod
    def create_easy_hard_negatives(self):
        pass

    @abstractmethod
    def shuffle_option_positions(self):
        pass

    def save(self):
        os.makedirs(DATA_DIR)
        self.dataset.to_json(DATA_DIR.joinpath(f"{self.task}.json"))


class BBHMultipleChoiceTask(BBHTask):

    def __init__(self, task: str):
        super(BBHMultipleChoiceTask, self).__init__(task)
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def option_str_to_options(self, option_str: str) -> List[str]:
        options = []
        for i, letter in enumerate(self.alphabet):
            start = option_str.find(f"({letter})")
            if start == -1:
                break
            else:
                if f"({self.alphabet[i+1]})" in option_str:
                    option = option_str[start+3:option_str.find(f"({self.alphabet[i+1]})")].strip()
                else:
                    option = option_str[start+3:].strip()
                options.append(option)
        return options

    def options_to_option_str(self, options: List[str]) -> str:
        option_str = ""
        for i, option in enumerate(options):
            option_str += f"({self.alphabet[i]}) {option}\n"
        return option_str.strip()

    def get_correct_option_str(self, input_str: str, target_letter: str):
        if len(target_letter) == 1:
            target_letter = f"({target_letter})"
        start = input_str.find(target_letter) + 3
        end = input_str.find("(", start)

        if end == -1:
            # if this is last option, take rest of the string as answer
            correct_option_str = input_str[start:].strip()
        else:
            # otherwise take string until the next option
            correct_option_str = input_str[start:end].strip()
        return correct_option_str

    def shuffle_option_positions(self):

        def shuffle_options(examples):
            for i, input_str in enumerate(examples["input"]):
                options_str = input_str[input_str.find("Options:\n") + 9:]
                options = self.option_str_to_options(options_str)
                correct_option_idx = self.alphabet.index(examples["target"][i][1])
                correct_option_str = self.get_correct_option_str(input_str, examples["target"][i])
                shuffled_options = options.copy()
                # shuffle until the correct answer letter is not the same as the original answer letter
                new_correct_opiton_idx = correct_option_idx
                while new_correct_opiton_idx == correct_option_idx:
                    np.random.shuffle(shuffled_options)
                    new_correct_opiton_idx = shuffled_options.index(correct_option_str)
                shuffled_options_str = self.options_to_option_str(shuffled_options)
                examples["input"][i] = input_str.replace(options_str, shuffled_options_str)
                examples["target"][i] = f"({self.alphabet[new_correct_opiton_idx]})"

            return examples

        self.dataset = self.dataset.map(shuffle_options, batched=True, ) # remove_columns=dataset.column_names

    def create_random_negatives(self):

        def random_negatives(examples):
            examples["random_negative"] = []
            for i, input_str in enumerate(examples["input"]):
                options_str = input_str[input_str.find("Options:\n") + 9:]
                options = self.option_str_to_options(options_str)
                alphabet_options = self.alphabet[:len(options)]
                alphabet_options = alphabet_options.replace(examples["target"][i][1], "")
                random_letter = np.random.choice(list(alphabet_options))
                examples["random_negative"].append(f"({random_letter})")
            return examples

        self.dataset = self.dataset.map(random_negatives, batched=True) # remove_columns=dataset.column_names

    def create_easy_hard_negatives(self, embedding_model="sentence-transformers/sentence-t5-xxl", embed_with_sentence=True):
        # model = SentenceTransformer("sentence-transformers/sentence-t5-xxl")
        # model = SentenceTransformer("sentence-transformers/paraphrase-albert-small-v2")
        model = SentenceTransformer(embedding_model)
        def easy_hard_negatives(examples):
            examples["easy_negative"] = []
            examples["hard_negative"] = []
            for i, input_str in enumerate(examples["input"]):
                options_str = input_str[input_str.find("Options:\n") + 9:]
                options = self.option_str_to_options(options_str)
                correct_option_idx = self.alphabet.index(examples["target"][i][1])
                if not embed_with_sentence:
                    # method 1: just look at option embeddings
                    sentences = options
                else:
                    # method 2: input + option
                    sentences = [input_str + f". The correct answer is {o}" for o in options]
                embs = model.encode(sentences)
                similarities = model.similarity(embs, embs)
                sorted_idx = similarities[correct_option_idx].argsort(descending=True)
                hard_idx = sorted_idx[1]  # most similar (0) is itself
                easy_idx = sorted_idx[-1]

                examples["easy_negative"].append(f"({self.alphabet[easy_idx]})")
                examples["hard_negative"].append(f"({self.alphabet[hard_idx]})")
            return examples

        self.dataset = self.dataset.map(easy_hard_negatives, batched=True)  # remove_columns=dataset.column_names


class BBHBinaryTask(BBHTask):

    def __init__(self, task: str):
        super(BBHBinaryTask, self).__init__(task)
        self.options = list(set(self.dataset["target"]))

    def shuffle_option_positions(self):
        """Nothing can be done for these"""
        pass

    def create_random_negatives(self):
        """ since there are only two options, negatives is the other option """
        def random_negatives(examples):
            examples["random_negative"] = []
            for i, target in enumerate(examples["target"]):
                other_option = [o for o in self.options if o!=target][0]
                examples["random_negative"].append(other_option)
            return examples

        self.dataset = self.dataset.map(random_negatives, batched=True) # remove_columns=dataset.column_names

    def create_easy_hard_negatives(self):
        """ There is only two choices, no way to differentiate easy vs hard negatives """
        self.create_random_negatives()
        self.dataset.add_column("easy_negative", self.dataset["random_negative"])
        self.dataset.add_column("hard_negative", self.dataset["random_negative"])


class BBHNumericalTask(BBHTask):

    def __init__(self, task: str, negative_strategy: str="plus_minus"):
        super(BBHNumericalTask, self).__init__(task)
        targets = np.array(self.dataset["target"]).astype(int)
        self.target_mean = targets.mean()
        self.target_std = targets.std()
        assert negative_strategy in ["plus_minus", "perturb_input"]
        self.negative_strategy = negative_strategy

    def shuffle_option_positions(self):
        """Nothing can be done for these"""
        pass

    @ staticmethod
    def random_negatives_plus_minus(examples, diff_mean, diff_std, field_name):
        examples[field_name] = []
        for i, target in enumerate(examples["target"]):
            diff = 0
            while diff == 0:
                diff = int(np.random.normal(diff_mean, diff_std, 1))
            examples[field_name].append(target + diff)
        return examples

    def create_random_negatives(self):
        """ since there are only two options, negatives is the other option """
        if self.negative_strategy == "plus_minus":
            self.dataset = self.dataset.map(
                partial(self.random_negatives_plus_minus, diff_mean=0, diff_std=self.target_std, field_name="random_negative"),
                batched=True) # remove_columns=dataset.column_names
        elif self.negative_strategy == "perturb_input":
            pass # this should be implemented by  child class

    def create_easy_hard_negatives(self):
        """
        option 1: plus minus small number to target object
        option 2: if there are other numbers that appeared in the prompt, use those as target
        """
        if self.negative_strategy == "plus_minus":
            self.dataset = self.dataset.map(
                partial(self.random_negatives_plus_minus, diff_mean=0, diff_std=self.target_std,  field_name="easy_negative"),
                batched=True)
            self.dataset = self.dataset.map(
                partial(self.random_negatives_plus_minus, diff_mean=0, diff_std=1,  field_name="hard_negative"),
                batched=True)


class DykeLanguagesTask(BBHTask):

    def __init__(self, task: str):
        super(DykeLanguagesTask, self).__init__(task)

    def shuffle_option_positions(self):
        pass

    def create_random_negatives(self):

        def random_negatives(examples):
            examples["random_negative"] = []
            for i, input_str in enumerate(examples["input"]):
                target = examples["target"][i]

                if np.random.rand()< 1/3 and len (target.split()) > 1:
                    # with 1/3 chance, randomly delete a symbol from target
                    target_symbols = target.split()
                    del_idx = np.random.randint(len(target_symbols))
                    new_target = " ".join([s for i,s in enumerate(target_symbols) if i != del_idx])

                elif np.random.rand() < 2/3 and len(set(target.split())) > 1:
                    # with 1/3 chance, randomly swap symbols
                    target_symbols = target.split()
                    new_symbols = target_symbols.copy()
                    new_target = target
                    while new_target == target:
                        swap_indices = np.random.choice(len(target_symbols), size=2, replace=False)
                        new_symbols[swap_indices[0]], new_symbols[swap_indices[1]] = new_symbols[swap_indices[1]], new_symbols[swap_indices[0]]
                        new_target = " ".join(new_symbols)

                else:  # randomly add a symbol that appears in input
                    all_symbols = []
                    for l, r in {"(":")", "<":">", "[":"]", "{":"}"}.items():
                        if l in input_str:
                            all_symbols.append(r)

                    target_symbols = examples["target"][i].split()
                    insert_idx = np.random.randint(len(target_symbols) + 1)
                    target_symbols.insert(insert_idx, np.random.choice(all_symbols))
                    new_target = " ".join(target_symbols)

                examples["random_negative"].append(new_target)
            return examples

        self.dataset = self.dataset.map(random_negatives, batched=True) # remove_columns=dataset.column_names

    def create_easy_hard_negatives(self):
        pass


class WordSortingTask(BBHTask):

    def __init__(self, task: str):
        super(WordSortingTask, self).__init__(task)

    def shuffle_option_positions(self):
        pass

    @staticmethod
    def random_negative_swaps(examples, k, field_name):
        examples[field_name] = []
        for i, target in enumerate(examples["target"]):
            new_target = target
            while new_target == target:
                new_target_list = target.split()
                iter_time = k if isinstance(k, int) else max(int(k*len(new_target_list)), 1)
                for _ in range(iter_time):
                    swap_indices = np.random.choice(len(new_target_list), size=2, replace=False)
                    new_target_list[swap_indices[0]], new_target_list[swap_indices[1]] = new_target_list[swap_indices[1]], new_target_list[swap_indices[0]]
                new_target = " ".join(new_target_list)
            examples[field_name].append(new_target)
        return examples

    def create_random_negatives(self):
        """ since there are only two options, negatives is the other option """
        self.dataset = self.dataset.map(
            partial(self.random_negative_swaps, k=1, field_name="random_negative"),
            batched=True) # remove_columns=dataset.column_names

    def create_easy_hard_negatives(self):
        # swap 30% of the pairs (each time swaps two randomly chosen index)
        self.dataset = self.dataset.map(
            partial(self.random_negative_swaps, k=0.3, field_name="easy_negative"),
            batched=True)  # remove_columns=dataset.column_names

        self.dataset = self.dataset.map(
            partial(self.random_negative_swaps, k=1, field_name="hard_negative"),
            batched=True)  # remove_columns=dataset.column_names


def to_camel_case(snake_str: str) -> str:
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def create_pairwise_bbh():
    for task in all_tasks:
        d = load_bbh_with_negatives(task)
        d.save()


def load_bbh_with_negatives(task: str, easy_hard_negatives=True) -> BBHTask:
    """
    Factory method to load a BBH dataset with negatives in memory

    Args:
        task (str): the bbh task string name
        easy_hard_negatives: whether to create easy/hard negatives when possible
    """
    if task in multiple_choice_tasks:
        d = BBHMultipleChoiceTask(task)
        d.shuffle_option_positions()
    elif task in binary_tasks:
        d = BBHBinaryTask(task)
    elif task in numerical_tasks:
        d = BBHNumericalTask(task)
    else:  # custom tasks
        try:
            d = globals()[to_camel_case(task)](task)
        except Exception as e:
            raise NotImplementedError(f"Task entered ({task}) is not loaded correctly: {e}")

    d.create_random_negatives()
    if easy_hard_negatives:
        d.create_easy_hard_negatives()
    return d

def generate_bbh_prompts(easy_hard_negatives: bool=True):
    """
    For BBH tasks, generate three type of prompts (pointwise {needs work}, pairwise, and pointwise).
    For each of the evaluation type, we also generate multiple versions depending on whether the negatives/responses
    are easy or hard or random. Output dataset.json individually, can be combined afterwards

    Args:
        rubric_dir (str): Directories of BBH rubrics
        out_dir (str): Output directory
        easy_hard_negatives (bool): whether to use heuristics to create easy/hard negatives
            (as opposed to random negatives). Currently all easy/hard negatives
    Output:
        save dataset json files to data/prompts/bbh
    """
    binary_dfs = {"target": [], "random": [], "easy": [], "hard": []}
    for task in tqdm(all_tasks):
        df = load_bbh_with_negatives(task, easy_hard_negatives=easy_hard_negatives).dataset

        ## generate binary evaluation prompt with BBH
        for response_col in ["target", "random_negative", "easy_negative", "hard_negative"]:
            if response_col in df.column_names:
                prompt_df = generate_prompt_df(
                    df=df,
                    rubric_load_path=os.path.join(RUBRICS_DIR, task, 'binary_rubric.json'),
                    evaluation_mode='binary',
                    input_data_column='input',
                    response_column=response_col,
                )
                binary_dfs[response_col.replace("_negative", "")].append(prompt_df)

    for negative, dfs in binary_dfs.items():
        datasets.concatenate_datasets(dfs).to_json(os.path.join(DATA_DIR, f'bbh_binary_prompts_{negative}.json'))


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    ## Use this to create all negatives and save locally (but can be skipped)
    # create_pairwise_bbh()

    generate_bbh_prompts(
        easy_hard_negatives=False
    )
