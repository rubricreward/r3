import argparse
import os
import json
import logging
import re
from collections import defaultdict

import numpy as np
from datasets import load_dataset
import pandas as pd
from scipy.stats import kendalltau, pearsonr

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(CUR_DIR)))

VALID_EVAL_DATASETS = [
    "RM-Bench", "reward-bench",
    "FeedbackBench", "MMLU-STEM", "BBH",
    "XSUM_faithfulness", "XSUM_coherence", "XSUM_relevance"
]

REWARDBENCH_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

REWARDBENCH_SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

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

def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Helper function for immediately logging RewardBench scores.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores

def evaluate_bbh(response_path, output_path):
    data = []
    parsed_scores = []
    failed_count = 0
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))

    for item in data:
        try:
            json_part = item["response"]
            score_value = extract_score(json_part)
            parsed_scores.append({'id': item['id'], 'answer': score_value})
        except (IndexError, json.JSONDecodeError, KeyError, TypeError) as e:
            failed_count += 1
            logging.info(f"Parsing failed for id {item.get('id', 'unknown')}: {e}")
            parsed_scores.append({'id': item['id'], 'answer': "Error"})

    parsed_df = pd.DataFrame(parsed_scores)
    dataset = load_dataset("rubricreward/R3-eval-BBH", split="train")
    gt_df = dataset.to_pandas()[['id', 'score']]

    # Merge predictions with ground truth on 'id'
    merged = pd.merge(parsed_df, gt_df, on='id')

    # Calculate match
    merged['match'] = merged['score'] == merged['answer']
    correct_count = merged['match'].sum()

    summary = {
        "Accuracy (Valid Only)": round(correct_count / len(merged[merged['answer'] != "Error"]), 4),
        "Raw Accuracy (Including Errors)": round(correct_count / len(merged), 4),
        "Failed Parses": int(failed_count)
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Total instances: {len(data)}")
    logging.info(f"Successfully parsed: {len(data) - failed_count}")
    logging.info(f"Failed to parse: {failed_count}")

def evaluate_feedbackbench(response_path, output_path):
    data = []
    parsed_scores = []
    failed_count = 0
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))

    for item in data:
        try:
            json_part = item["response"]
            score = extract_score(json_part)
            score_value = int(score)
            parsed_scores.append({'id': item['id'], 'answer': score_value})
        except Exception as e:
            failed_count += 1
            logging.info(f"Parsing failed for id {item.get('id', 'unknown')}: {e}")
            parsed_scores.append({'id': item['id'], 'answer': "Error"})

    parsed_df = pd.DataFrame(parsed_scores)

    # Load ground truth
    dataset = load_dataset("rubricreward/R3-eval-FeedbackBench", split="train")
    gt_df = dataset.to_pandas()[['id', 'score']]

    # Merge on 'id'
    merged = pd.merge(parsed_df, gt_df, on='id')    

    # Filter out 'Error' values
    clean = merged[merged['answer'] != "Error"].copy()
    clean['answer'] = clean['answer'].astype(int)
    clean['score'] = clean['score'].astype(int)

    # Match for exact correctness
    clean['match'] = clean['score'] == clean['answer']
    correct_count = clean['match'].sum()
    total_count = len(clean)

    # Correlation metrics
    if total_count > 1:  # At least 2 points needed
        kendall_tau, _ = kendalltau(clean['score'], clean['answer'])
        pearson_corr, _ = pearsonr(clean['score'], clean['answer'])
    else:
        kendall_tau = None
        pearson_corr = None

    summary = {
        "Accuracy (Valid Only)": round(correct_count / total_count, 4) if total_count else "N/A",
        "Raw Accuracy (Including Errors)": round(correct_count / len(merged), 4),
        "Kendall_tau": round(kendall_tau, 4) if kendall_tau is not None else "N/A",
        "Pearson": round(pearson_corr, 4) if pearson_corr is not None else "N/A",
        "Failed Parses": int(failed_count)
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Total instances: {len(data)}")
    logging.info(f"Successfully parsed: {len(data) - failed_count}")
    logging.info(f"Failed to parse: {failed_count}")

def evaluate_mmlustem(response_path, output_path):
    data = []
    parsed_scores = []
    failed_count = 0
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))

    for item in data:
        try:
            json_part = item["response"]
            score_value = extract_score(json_part)
            parsed_scores.append({'id': item['id'], 'answer': score_value})
        except (IndexError, json.JSONDecodeError, KeyError, TypeError) as e:
            failed_count += 1
            logging.info(f"Parsing failed for id {item.get('id', 'unknown')}: {e}")
            parsed_scores.append({'id': item['id'], 'answer': "Error"})

    parsed_df = pd.DataFrame(parsed_scores)

    # Load ground truth
    dataset = load_dataset("rubricreward/R3-eval-MMLU-STEM", split="train")
    gt_df = dataset.to_pandas()[['id', 'score', 'subject']]

    # Merge on 'id'
    merged = pd.merge(parsed_df, gt_df, on='id')

    # Mark failed parses (i.e., missing or error answer)
    merged['parsed'] = merged['answer'] != "Error"
    
    subject_correct = defaultdict(int)
    subject_total = defaultdict(int)
    subject_failed = defaultdict(int)

    # Manually count per subject
    for _, row in merged.iterrows():
        subject = row['subject']
        subject_total[subject] += 1
        if not row['parsed']:
            subject_failed[subject] += 1
        elif str(row['score']).lower() == str(row['answer']).lower():
            subject_correct[subject] += 1

    # Build result
    subject_stats = []
    for subject in subject_total:
        total = subject_total[subject]
        correct = subject_correct[subject]
        failed = subject_failed[subject]
        accuracy = round(correct / total, 4) if total > 0 else 0.0

        subject_stats.append({
            "subject": subject,
            "correct": correct,
            "total": total,
            "failed": failed,
            "accuracy": accuracy
        })

    # Final summary
    summary = {
        "Overall Accuracy": round(sum(s["correct"] for s in subject_stats) / sum(s["total"] for s in subject_stats), 4),
        "Failed Parses": sum(s["failed"] for s in subject_stats),
        "Per-Subject Accuracy": sorted(subject_stats, key=lambda x: x["subject"])
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Total instances: {len(data)}")
    logging.info(f"Successfully parsed: {len(data) - failed_count}")
    logging.info(f"Failed to parse: {failed_count}")

def evaluate_rewardbench(response_path, output_path):
    data = []
    parsed_scores = []
    failed_count = 0
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))

    for item in data:
        try:
            json_part = item["response"]
            score_value = extract_score(json_part)
            parsed_scores.append({'id': item['id'], 'answer': score_value})
        except (IndexError, json.JSONDecodeError, KeyError, TypeError) as e:
            failed_count += 1
            logging.info(f"Parsing failed for id {item.get('id', 'unknown')}: {e}")
            parsed_scores.append({'id': item['id'], 'answer': "Error"})

    parsed_df = pd.DataFrame(parsed_scores)

    # Load ground truth
    dataset = load_dataset("rubricreward/R3-eval-reward-bench", split="train")
    gt_df = dataset.to_pandas()[['id', 'score']]
    parsed_df_sorted = parsed_df.sort_values(by='id').reset_index(drop=True)
    gt_df_sorted = gt_df.sort_values(by='id').reset_index(drop=True)

    # Concatenate side-by-side
    merged = pd.concat([parsed_df_sorted, gt_df_sorted[['score']]], axis=1)
    results = [1 if row['score'] == row['answer'] else 0 for _, row in merged.iterrows()]

    # Reward bench calculation eval
    dataset = load_dataset("allenai/reward-bench", split="filtered")
    subsets = dataset["subset"]
    all_cols = dataset.column_names
    keep_columns = ["text_chosen", "text_rejected", "prompt"]
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    accuracy = sum(results) / len(results)
    logging.info(f"Results: {accuracy}, on {len(results)} prompts")
    
    out_dataset = dataset.add_column("results", results)
    out_dataset = out_dataset.add_column("subsets", subsets)
    out_dataset = out_dataset.to_pandas()

    results_grouped = {}
    present_subsets = np.unique(out_dataset["subsets"])
    for subset in present_subsets:
        subset_dataset = out_dataset[out_dataset["subsets"] == subset]
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        logging.info(f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})")
        results_grouped[subset] = num_correct / num_total

    results_section = calculate_scores_per_section(REWARDBENCH_COUNTS, REWARDBENCH_SUBSET_MAPPING, results_grouped)
    results_section['Average'] = sum(results_section.values()) / len(results_section)
    logging.info(f"Results by Section: {results_section}")

    with open(output_path, "w") as f:
        json.dump(results_section, f, indent=4)

def evaluate_rmbench(response_path, output_path):
    data = []
    parsed_scores = []
    failed_count = 0
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))

    for item in data:
        try:
            json_part = item["response"]
            score_value = extract_score(json_part)
            parsed_scores.append({'id': item['id'], 'answer': score_value})
        except (IndexError, json.JSONDecodeError, KeyError, TypeError) as e:
            failed_count += 1
            logging.info(f"Parsing failed for id {item.get('id', 'unknown')}: {e}")
            parsed_scores.append({'id': item['id'], 'answer': "Error"})

    parsed_df = pd.DataFrame(parsed_scores)

    # Load ground truth
    dataset = load_dataset("rubricreward/R3-eval-RM-Bench", split="train")
    gt_df = dataset.to_pandas()[['id', 'score']]
    parsed_df_sorted = parsed_df.sort_values(by='id').reset_index(drop=True)
    gt_df_sorted = gt_df.sort_values(by='id').reset_index(drop=True)
    merged_df = pd.concat([parsed_df_sorted, gt_df_sorted[['score']]], axis=1)
    
    # Iterate per domain
    MATRIX_SIZE = 3
    # Group by domain, e.g., "chat", "math", etc.
    domain_results = {}
    for domain, domain_group in merged_df.groupby(merged_df["id"].apply(lambda fid: fid.split("/")[0].split("-")[0])):
        acc_matrix = np.zeros((3, 3), float)
        
        # Group by original ID within this domain
        for _, group in domain_group.groupby(domain_group["id"].apply(lambda fid: "-".join(fid.split("-")[:-2]))):
            for _, row in group.iterrows():
                parts = row["id"].split("-")
                i, j = int(parts[-2]), int(parts[-1])
                acc_matrix[i, j] += str(row["score"]).lower() == str(row["answer"]).lower()

        divide_by = len(domain_group) / (MATRIX_SIZE * MATRIX_SIZE)
        acc_matrix /= divide_by # Domain group is 9 * the number of domains
        # compute the hard,normal,easy accuracy (note this is adapted from RM-Bench official repo)
        # hard accuracy: the average of the upper-right triangle of the matrix
        # namely chosen responses with less fancy style compared to rejected responses with more fancy style
        upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count
        # normal accuracy: the average of the diagonal of the matrix
        # namely chosen responses with the same style compared to rejected responses with the same style
        normal_acc = np.mean(np.diag(acc_matrix))
        # easy accuracy: the average of the lower-left triangle of the matrix
        # namely chosen responses with more fancy style compared to rejected responses with less fancy style
        lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count
        
        domain_results[domain] = {
            "easy_acc": easy_acc,
            "normal_acc": normal_acc,
            "hard_acc": hard_acc,
        }
        
    # Calculate avg per domain
    res_per_domain = {}
    for domain in domain_results:
        res_per_domain[domain] = np.mean([
            domain_results[domain]['easy_acc'],
            domain_results[domain]['normal_acc'],
            domain_results[domain]['hard_acc']
        ])

    # Calculate avg per difficulty across all domains
    res_per_difficulty = {}
    for diff in ['easy_acc', 'normal_acc', 'hard_acc']:
        res_per_difficulty[diff] = np.mean([
            domain_results[domain][diff] for domain in domain_results
        ])

    # Calculate overall average accuracy
    total_avg_acc = np.mean(list(res_per_domain.values()))
    results_section = {
        "result_per_domain": res_per_domain,
        "result_per_difficulty": res_per_difficulty,
        "total_avg_acc": total_avg_acc
    }

    logging.info(results_section)
    with open(output_path, "w") as f:
        json.dump(results_section, f, indent=4)

def evaluate_xsum_faithfulness(response_path, output_path, split):
    data = []
    parsed_scores = []
    failed_count = 0
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))

    for item in data:
        try:
            json_part = item["response"]
            score_value = extract_score(json_part)
            parsed_scores.append({'id': item['id'], 'answer': score_value})
        except (IndexError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            failed_count += 1
            logging.info(f"Parsing failed for id {item.get('id', 'unknown')}: {e}")
            parsed_scores.append({'id': item['id'], 'answer': "Error"})

    parsed_df = pd.DataFrame(parsed_scores)

    # Load ground truth
    dataset = load_dataset("rubricreward/R3-eval-XSUM", split=split)
    gt_df = dataset.to_pandas()[['id', 'score']]
    gt_df['score'] = np.where(gt_df['score'] == 1, 'true', 'false')
    # Merge predictions with ground truth on 'id'
    merged = pd.merge(parsed_df, gt_df, on='id')

    # Calculate match
    merged['match'] = merged['score'].str.lower() == merged['answer'].str.lower()
    correct_count = merged['match'].sum()

    summary = {
        "Accuracy (Valid Only)": round(correct_count / len(merged[merged['answer'] != "Error"]), 4),
        "Raw Accuracy (Including Errors)": round(correct_count / len(merged), 4),
        "Failed Parses": int(failed_count)
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Total instances: {len(data)}")
    logging.info(f"Successfully parsed: {len(data) - failed_count}")
    logging.info(f"Failed to parse: {failed_count}")

def evaluate_xsum(response_path, output_path, split):
    data = []
    parsed_scores = []
    failed_count = 0
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))

    for item in data:
        try:
            json_part = item["response"]
            score = extract_score(json_part)
            score_value = int(score)
            parsed_scores.append({'id': item['id'], 'answer': score_value})
        except (IndexError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            failed_count += 1
            logging.info(f"Parsing failed for id {item.get('id', 'unknown')}: {e}")
            parsed_scores.append({'id': item['id'], 'answer': "Error"})

    parsed_df = pd.DataFrame(parsed_scores)

    # Load ground truth
    dataset = load_dataset("rubricreward/R3-eval-XSUM", split=split)
    gt_df = dataset.to_pandas()[['id', 'score']]

    # Merge on 'id'
    merged = pd.merge(parsed_df, gt_df, on='id')    

    # Filter out 'Error' values
    clean = merged[merged['answer'] != "Error"].copy()
    clean['answer'] = clean['answer'].astype(int)
    clean['score'] = clean['score'].astype(int)

    # Match for exact correctness
    clean['match'] = clean['score'] == clean['answer']
    correct_count = clean['match'].sum()
    total_count = len(clean)

    # Correlation metrics
    if total_count > 1:  # At least 2 points needed
        kendall_tau, _ = kendalltau(clean['score'], clean['answer'])
        pearson_corr, _ = pearsonr(clean['score'], clean['answer'])
    else:
        kendall_tau = None
        pearson_corr = None

    summary = {
        "Accuracy (Valid Only)": round(correct_count / total_count, 4) if total_count else "N/A",
        "Raw Accuracy (Including Errors)": round(correct_count / len(merged), 4),
        "Kendall_tau": round(kendall_tau, 4) if kendall_tau is not None else "N/A",
        "Pearson": round(pearson_corr, 4) if pearson_corr is not None else "N/A",
        "Failed Parses": int(failed_count)
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    logging.info(f"Total instances: {len(data)}")
    logging.info(f"Successfully parsed: {len(data) - failed_count}")
    logging.info(f"Failed to parse: {failed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on R3 benchmark datasets.")
    parser.add_argument(
        '--dataset_names',
        '-d', type=str, default="all", 
        help="List of dataset to be evaluated upon, separated by comma(s). `all` means infer on all."
    )
    parser.add_argument(
        "--response_folder_path",
        type=str,
        default=CUR_DIR,
        help="Path to the response folder.",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default='output_eval',
        help="File path to save evaluation results.",
    )
    args = parser.parse_args()
    
    dataset_names = args.dataset_names.strip()
    eval_dataset_list = []
    if dataset_names == "all":
        eval_dataset_list = VALID_EVAL_DATASETS
    else:
        dataset_name_list = dataset_names.split(",")
        for dataset_name in dataset_name_list:
            if dataset_name in VALID_EVAL_DATASETS:
                eval_dataset_list.append(dataset_name)
            else:
                logging.warning(f"Unrecognized evaluation dataset named `{dataset_name}`, skipping ...")

    if len(eval_dataset_list) == 0:
        raise ValueError("Evaluation datasets cannot be empty!")
    
    for dataset_name in eval_dataset_list:
        response_path = os.path.join(args.response_folder_path, dataset_name)
        output_path = os.path.join(args.output_folder_path, dataset_name)
        if os.path.exists(response_path):
            if dataset_name == "RM-Bench":
                evaluate_rmbench(response_path, output_path=output_path)
            elif dataset_name == "reward-bench": # 2985 pairs
                evaluate_rewardbench(response_path, output_path=output_path)
            elif dataset_name == "FeedbackBench": # 1000 instances
                evaluate_feedbackbench(response_path, output_path=output_path)
            elif dataset_name == "MMLU-STEM": # 6306 instances
                evaluate_mmlustem(response_path, output_path=output_path)
            elif dataset_name == "BBH": # 1000 instances
                evaluate_bbh(response_path, output_path=output_path)    
            elif dataset_name.startswith("XSUM"):
                split = dataset_name.split("_")[-1]
                if dataset_name == "XSUM_faithfulness":
                    evaluate_xsum_faithfulness(args.response_path, output_path=output_path, split=split)
                else:
                    evaluate_xsum(response_path, output_path=output_path, split=split)
            else:
                logging.warning(f"Unrecognized evaluation dataset named `{dataset_name}`, skipping ...")
            
            logging.info(f"Successfully evaluated `{dataset_name}`, saving to `{output_path}`")
        else:
            raise ValueError(f"Response path `{response_path}` does not exist!")
