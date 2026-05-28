import json
import argparse
import os
from datetime import datetime

LABELS = ['A', 'B', 'C', 'D']


def load_json_file(file_path):
    """Loads a JSON file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def has_nonempty_final_answer(item):
    """
    Return True if item has a final_answer field that is a non-empty string
    (after stripping whitespace).
    This is used to decide whether an ID is considered 'present' in a result file.
    """
    fa = item.get('final_answer', None)
    return isinstance(fa, str) and fa.strip() != ''


def normalize_answer(answer):
    """
    Normalize the final choice:
    - If answer is 'A', 'B', or 'D', keep it.
    - Otherwise treat it as 'C'.
    """
    if answer in ['A', 'B', 'D']:
        return answer
    return 'C'


def get_pair_prediction(answer1, answer2):
    """
    Derive a single predicted label (A/B/C/D) from the pair (answer1, answer2)
    according to the rules used in evaluation.

    answer1, answer2 are already normalized to A/B/C/D or possibly None.
    """
    # Build a set ignoring Nones
    answers_set = {a for a in (answer1, answer2) if a is not None}

    # D: one run says A, the other says B
    if answers_set == {'A', 'B'}:
        return 'D'

    # A: one run says A, the other says C (or single A)
    if answers_set in [{'A', 'C'}, {'A'}]:
        return 'A'

    # B: one run says B, the other says C (or single B)
    if answers_set in [{'B', 'C'}, {'B'}]:
        return 'B'

    # C: both runs say C or one run is C and the other is missing/None
    if answer1 == 'C' and (answer2 == 'C' or answer2 is None):
        return 'C'
    if answer1 is None and answer2 is None:
        return 'C'

    # Default to C for any other combination
    return 'C'


def evaluate_instance(id_, instance1, instance2, ground_truth):
    """
    Evaluates two instances with the same 'id':
    - Extract final choices from both files
    - Normalize them
    - Derive a single predicted label
    - Return (is_correct, predicted_label, normalized_answer1, normalized_answer2)
    """
    # Get answers for instance1
    try:
        default1 = 'C' if 'No final answer found in the text.' in instance1.get('final_answer', '') else None
        answer1 = instance1.get('final_choice', default1)
    except Exception:
        answer1 = None

    # Get answers for instance2
    try:
        default2 = 'C' if 'No final answer found in the text.' in instance2.get('final_answer', '') else None
        answer2 = instance2.get('final_choice', default2)
    except Exception:
        answer2 = None

    answer1 = normalize_answer(answer1) if answer1 is not None else None
    answer2 = normalize_answer(answer2) if answer2 is not None else None

    pred_label = get_pair_prediction(answer1, answer2)
    is_correct = (pred_label == ground_truth)

    return is_correct, pred_label, answer1, answer2


def init_confusion_matrix():
    return {gt: {pred: 0 for pred in LABELS} for gt in LABELS}


def compute_metrics(confusion):
    metrics = {}
    total_correct = 0
    total_instances = 0

    for gt in LABELS:
        for pred in LABELS:
            count = confusion[gt][pred]
            total_instances += count
            if gt == pred:
                total_correct += count

    accuracy = total_correct / total_instances if total_instances > 0 else 0.0

    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gt][label] for gt in LABELS if gt != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        }

    macro_precision = sum(per_class[l]['precision'] for l in LABELS) / len(LABELS)
    macro_recall = sum(per_class[l]['recall'] for l in LABELS) / len(LABELS)

    metrics['accuracy'] = accuracy
    metrics['per_class'] = per_class
    metrics['macro_precision'] = macro_precision
    metrics['macro_recall'] = macro_recall
    metrics['total_instances'] = total_instances
    metrics['total_correct'] = total_correct

    return metrics


def format_confusion_matrix(confusion):
    header = [' '] + LABELS
    lines = ['\t'.join(header)]
    for gt in LABELS:
        row = [gt] + [str(confusion[gt][pred]) for pred in LABELS]
        lines.append('\t'.join(row))
    return '\n'.join(lines)


# -------------------- TOKEN STATS HELPERS --------------------


def aggregate_token_stats(values_by_key):
    """
    values_by_key: dict like
      {
        'prompt_tokens': [(val, id), ...],
        'completion_tokens': [...],
        'total_tokens': [...]
      }
    Returns dict:
      {
        'prompt_tokens': {'min': ..., 'min_id': ..., 'max': ..., 'max_id': ..., 'avg': ...},
        ...
      }
    If no data for a key, that entry is None.
    """
    out = {}
    for key, arr in values_by_key.items():
        if not arr:
            out[key] = None
            continue
        vals = [v for v, _ in arr]
        min_val, min_id = min(arr, key=lambda x: x[0])
        max_val, max_id = max(arr, key=lambda x: x[0])
        avg_val = sum(vals) / len(vals)
        out[key] = {
            "min": min_val,
            "min_id": min_id,
            "max": max_val,
            "max_id": max_id,
            "avg": avg_val
        }
    return out


def compute_token_stats_from_items(items, id_filter=None):
    """
    Compute per-file token stats.

    items: list of JSON items (each with id and token_usage)
    id_filter: optional set/list of ids to restrict to. If None, use all.
    """
    values_by_key = {
        'prompt_tokens': [],
        'completion_tokens': [],
        'total_tokens': []
    }

    for it in items:
        id_ = it.get('id', 'UNKNOWN_ID')
        if id_filter is not None and id_ not in id_filter:
            continue

        usage = it.get('token_usage') or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        values_by_key['prompt_tokens'].append((usage.get("prompt_tokens", 0), id_))
        values_by_key['completion_tokens'].append((usage.get("completion_tokens", 0), id_))
        values_by_key['total_tokens'].append((usage.get("total_tokens", 0), id_))

    return aggregate_token_stats(values_by_key)


def compute_combined_token_stats(valid_ids, file1_map, file2_map, file3_map):
    """
    For each id in valid_ids, sum token_usage across:
      - file1_map[id]
      - file2_map[id]
      - file3_map[id] (if present)
    Then aggregate min/max/avg over these per-id totals.
    """
    values_by_key = {
        'prompt_tokens': [],
        'completion_tokens': [],
        'total_tokens': []
    }

    for id_ in valid_ids:
        total_prompt = 0
        total_completion = 0
        total_total = 0

        for m in (file1_map, file2_map, file3_map):
            if m is None:
                continue
            item = m.get(id_)
            if item is None:
                continue
            usage = item.get('token_usage') or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)
            total_total += usage.get("total_tokens", 0)

        values_by_key['prompt_tokens'].append((total_prompt, id_))
        values_by_key['completion_tokens'].append((total_completion, id_))
        values_by_key['total_tokens'].append((total_total, id_))

    return aggregate_token_stats(values_by_key)


def print_token_section(name, stats):
    """
    Print one block of token stats to stdout.
    """
    print(f"\n{name}:")
    if not stats:
        print("  No token usage data found.")
        return

    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        s = stats.get(key)
        if s is None:
            print(f"  {key}: no data")
            continue
        print(f"  {key}:")
        print(f"    min: {s['min']} (id={s['min_id']})")
        print(f"    max: {s['max']} (id={s['max_id']})")
        print(f"    avg: {s['avg']:.2f}")


def append_token_section(lines, name, stats):
    """
    Append one block of token stats to the report lines list.
    """
    lines.append(name + ":")
    if not stats:
        lines.append("  No token usage data found.")
        lines.append("")
        return

    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        s = stats.get(key)
        if s is None:
            lines.append(f"  {key}: no data")
            continue
        lines.append(f"  {key}:")
        lines.append(f"    min: {s['min']} (id={s['min_id']})")
        lines.append(f"    max: {s['max']} (id={s['max_id']})")
        lines.append(f"    avg: {s['avg']:.2f}")
    lines.append("")


# -------------------- MAIN EVAL --------------------


def evaluate_files(dataset_name, model_name):
    file1_path = f'./results/{dataset_name}/{dataset_name}_{model_name}_search_negation_True.json'
    file2_path = f'./results/{dataset_name}/{dataset_name}_{model_name}_search_negation_False.json'
    file3_path = f'./results/{dataset_name}/{dataset_name}_{model_name}_trans_decompose_no_negation.json'
    dev_path = f'./data/{dataset_name}/dev.json'

    file1 = load_json_file(file1_path)
    file2 = load_json_file(file2_path)

    # handling trans_decompose file for their token usage
    try:
        file3 = load_json_file(file3_path)
    except Exception as e:
        print(f"Warning: could not load trans_decompose file at {file3_path}: {e}")
        file3 = []

    # Load dev.json and collect all distinct IDs (the "true" set of true dataset IDs for one benchmark dataset)
    dev_ids = set()
    try:
        dev_data = load_json_file(dev_path)
        for item in dev_data:
            id_ = item.get('id', None)
            if id_ is not None:
                dev_ids.add(id_)
    except Exception as e:
        print(f"Warning: could not load dev file at {dev_path}: {e}")
        dev_data = []
        dev_ids = set()

    file1_map = {}
    file2_map = {}
    file3_map = {}

    for item in file1:
        try:
            file1_map[item['id']] = item
        except Exception:
            print(f"Error: 'id' not found or invalid in file1 item: {item}")
    for item in file2:
        try:
            file2_map[item['id']] = item
        except Exception:
            print(f"Error: 'id' not found or invalid in file2 item: {item}")
    for item in file3:
        try:
            file3_map[item['id']] = item
        except Exception:
            print(f"Error: 'id' not found or invalid in file3 item: {item}")

    # IDs that have a non-empty final_answer in each file (for True/False only)
    ids_with_fa_true = {id_ for id_, it in file1_map.items() if has_nonempty_final_answer(it)}
    ids_with_fa_false = {id_ for id_, it in file2_map.items() if has_nonempty_final_answer(it)}

    # an ID in dev is "not missing" only if there is a corresponding item with non-empty final_answer in the relevant file(s).
    if dev_ids:
        missing_in_true = sorted(dev_ids - ids_with_fa_true)
        missing_in_false = sorted(dev_ids - ids_with_fa_false)
        missing_in_either = sorted(dev_ids - (ids_with_fa_true & ids_with_fa_false))
    else:
        missing_in_true = []
        missing_in_false = []
        missing_in_either = []

    # IDs we will  evaluate:
    # must be in dev . ids and have non-empty final_answer in BOTH Negation_True and Negation_False files.
    if dev_ids:
        valid_ids = sorted(dev_ids & ids_with_fa_true & ids_with_fa_false)
    else:
        valid_ids = sorted(ids_with_fa_true & ids_with_fa_false)

    total_instances = 0
    correct_instances = 0

    confusion = init_confusion_matrix()
    error_ids = []
    correct_ids = []

    # to be collecting first example per (gt, pred) error cell
    error_examples = {}

    for id_ in valid_ids:
        try:
            instance1 = file1_map.get(id_)
            instance2 = file2_map.get(id_)

            if instance1 is None or instance2 is None:
                continue

            ground_truth = instance1.get('ground_truth')

            if ground_truth and ground_truth in LABELS:
                total_instances += 1

                is_correct, pred_label, norm_answer1, norm_answer2 = evaluate_instance(
                    id_, instance1, instance2, ground_truth
                )

                # Update confusion matrix
                if pred_label in LABELS:
                    confusion[ground_truth][pred_label] += 1

                if is_correct:
                    correct_instances += 1
                    correct_ids.append(id_)
                else:
                    error_ids.append(id_)
                    key = (ground_truth, pred_label)
                    # Save first example for this non-diagonal cell
                    if ground_truth != pred_label and key not in error_examples:
                        error_examples[key] = {
                            'id': id_,
                            'instance1': instance1,
                            'instance2': instance2,
                            'ground_truth': ground_truth,
                            'predicted_label': pred_label,
                            'norm_answer1': norm_answer1,
                            'norm_answer2': norm_answer2,
                        }

        except Exception as e:
            print(f"Error processing instance with ID {id_}: {str(e)}")
            error_ids.append(id_)

    accuracy = correct_instances / total_instances if total_instances > 0 else 0.0

    # computing precision/recall and other metrics from confusion matrix
    metrics = compute_metrics(confusion)

    # time info for report filename
    now = datetime.now()
    today_date = now.strftime('%Y%m%d')
    today_time = now.strftime('%H%M%S')

    # TOKEN STATS (computed over valid_ids) 
    id_filter = set(valid_ids) if valid_ids else None

    token_stats_true = compute_token_stats_from_items(file1, id_filter=id_filter)
    token_stats_false = compute_token_stats_from_items(file2, id_filter=id_filter)
    token_stats_trans = compute_token_stats_from_items(file3, id_filter=id_filter) if file3 else None
    token_stats_combined = compute_combined_token_stats(valid_ids, file1_map, file2_map, file3_map) if valid_ids else None

    # PRINT TO STDOUT (including missing IDs first) 
    print(f"Dataset name: {dataset_name}")
    print(f"Model name: {model_name}")
    print(f"Evaluation date: {today_date} {today_time}")

    if dev_ids:
        print(f"\nTotal distinct IDs in dev: {len(dev_ids)}")
        print(f"Missing (no non-empty final_answer) in search_negation_True: {len(missing_in_true)}")
        print(f"Missing (no non-empty final_answer) in search_negation_False: {len(missing_in_false)}")
        print(f"Missing in at least one of the two files (under this definition): {len(missing_in_either)}")
        if missing_in_either:
            print("Example missing IDs (up to 20):", missing_in_either[:20])
    else:
        print("\nWarning: dev.json not available or no IDs found; missing-ID analysis skipped.")

    print(f"\nTotal evaluated instances (dev IDs with non-empty final_answer in both files): {total_instances}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nToken Usage Statistics (restricted to evaluated IDs):")
    print_token_section("search_negation_True", token_stats_true)
    print_token_section("search_negation_False", token_stats_false)
    if token_stats_trans is not None:
        print_token_section("trans_decompose_no_negation", token_stats_trans)
    if token_stats_combined is not None:
        print_token_section("Combined_per_ID_across_all_stages", token_stats_combined)

    print("\nConfusion Matrix (rows = ground truth, cols = prediction):")
    print(format_confusion_matrix(confusion))

    print("\nPer-class metrics:")
    for label in LABELS:
        m = metrics['per_class'][label]
        print(
            f"Label {label}: "
            f"Precision={m['precision']:.2%}, "
            f"Recall={m['recall']:.2%}, "
            f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}"
        )

    print(
        f"\nMacro Precision: {metrics['macro_precision']:.2%}, "
        f"Macro Recall: {metrics['macro_recall']:.2%}"
    )

    print("\nError ids:", error_ids)
    print("Correct ids:", correct_ids)

    # Print example errors per non-zero wrong cell, including reasoning_step,
    # translated_context, and normalized_context
    if error_examples:
        print("\nExample errors per (ground_truth, prediction) cell:")
        for (gt, pred), info in sorted(error_examples.items()):
            inst1 = info['instance1']
            inst2 = info['instance2']
            print(f"\nGround truth: {gt}, Predicted: {pred}")
            print(f"  id: {info['id']}")
            print(f"  original_context: {inst1.get('original_context', '')}")
            print(f"  question: {inst1.get('question', '')}")
            print(f"  ground_truth: {info['ground_truth']}")
            print(f"  predicted_label: {info['predicted_label']}")
            print(f"  search_negation_True final_choice: {inst1.get('final_choice', '')}")
            print(f"  search_negation_True final_answer: {inst1.get('final_answer', '')}")
            print(f"  search_negation_True reasoning_step: {inst1.get('reasoning_step', '')}")
            print(f"  search_negation_True translated_context: {inst1.get('translated_context', '')}")
            print(f"  search_negation_True normalized_context: {inst1.get('normalized_context', '')}")
            print(f"  search_negation_False final_choice: {inst2.get('final_choice', '')}")
            print(f"  search_negation_False final_answer: {inst2.get('final_answer', '')}")
            print(f"  search_negation_False reasoning_step: {inst2.get('reasoning_step', '')}")
            print(f"  search_negation_False translated_context: {inst2.get('translated_context', '')}")
            print(f"  search_negation_False normalized_context: {inst2.get('normalized_context', '')}")
            print(f"  normalized answers: (True_run={info['norm_answer1']}, False_run={info['norm_answer2']})")

    # BUILD REPORT TEXT FOR FILE 
    lines = []
    lines.append(f"Dataset name: {dataset_name}")
    lines.append(f"Model name: {model_name}")
    lines.append(f"Date: {today_date}")
    lines.append(f"Time: {today_time}")
    lines.append("")

    if dev_ids:
        lines.append(f"Total distinct IDs in dev: {len(dev_ids)}")
        lines.append(f"Missing (no non-empty final_answer) in search_negation_True: {len(missing_in_true)}")
        if missing_in_true:
            lines.append("IDs missing in search_negation_True:")
            lines.append(','.join(missing_in_true))
        lines.append("")
        lines.append(f"Missing (no non-empty final_answer) in search_negation_False: {len(missing_in_false)}")
        if missing_in_false:
            lines.append("IDs missing in search_negation_False:")
            lines.append(','.join(missing_in_false))
        lines.append("")
        lines.append(
            f"Missing in at least one of the two files (under this definition): {len(missing_in_either)}"
        )
        if missing_in_either:
            lines.append("IDs missing in at least one of the two files:")
            lines.append(','.join(missing_in_either))
        lines.append("")
    else:
        lines.append("dev.json not available or contained no IDs; missing-ID analysis skipped.")
        lines.append("")

    lines.append(f"Total evaluated instances: {total_instances}")
    lines.append(f"Accuracy: {accuracy:.4f}")
    lines.append("")

    # TOKEN STATS IN REPORT FILE 
    lines.append("Token Usage Statistics (restricted to evaluated IDs):")
    append_token_section(lines, "search_negation_True", token_stats_true)
    append_token_section(lines, "search_negation_False", token_stats_false)
    if token_stats_trans is not None:
        append_token_section(lines, "trans_decompose_no_negation", token_stats_trans)
    if token_stats_combined is not None:
        append_token_section(lines, "Combined_per_ID_across_all_stages", token_stats_combined)

    lines.append("Confusion Matrix (rows = ground truth, cols = prediction):")
    lines.append(format_confusion_matrix(confusion))
    lines.append("")

    lines.append("Per-class metrics:")
    for label in LABELS:
        m = metrics['per_class'][label]
        lines.append(
            f"Label {label}: "
            f"Precision={m['precision']:.4f}, "
            f"Recall={m['recall']:.4f}, "
            f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}"
        )
    lines.append("")
    lines.append(
        f"Macro Precision: {metrics['macro_precision']:.4f}, "
        f"Macro Recall: {metrics['macro_recall']:.4f}"
    )
    lines.append("")

    lines.append("Error ids:")
    lines.append(','.join(error_ids))
    lines.append("")
    lines.append("Correct ids:")
    lines.append(','.join(correct_ids))
    lines.append("")

    if error_examples:
        lines.append("")
        lines.append("Example errors per (ground_truth, prediction) cell:")
        for (gt, pred), info in sorted(error_examples.items()):
            inst1 = info['instance1']
            inst2 = info['instance2']
            lines.append(f"Ground truth: {gt}, Predicted: {pred}")
            lines.append(f"  id: {info['id']}")
            lines.append(f"  original_context: {inst1.get('original_context', '')}")
            lines.append(f"  question: {inst1.get('question', '')}")
            lines.append(f"  ground_truth: {info['ground_truth']}")
            lines.append(f"  predicted_label: {info['predicted_label']}")
            lines.append(f"  search_negation_True final_choice: {inst1.get('final_choice', '')}")
            lines.append(f"  search_negation_True final_answer: {inst1.get('final_answer', '')}")
            lines.append(f"  search_negation_True reasoning_step: {inst1.get('reasoning_step', '')}")
            lines.append(f"  search_negation_True translated_context: {inst1.get('translated_context', '')}")
            lines.append(f"  search_negation_True normalized_context: {inst1.get('normalized_context', '')}")
            lines.append(f"  search_negation_False final_choice: {inst2.get('final_choice', '')}")
            lines.append(f"  search_negation_False final_answer: {inst2.get('final_answer', '')}")
            lines.append(f"  search_negation_False reasoning_step: {inst2.get('reasoning_step', '')}")
            lines.append(f"  search_negation_False translated_context: {inst2.get('translated_context', '')}")
            lines.append(f"  search_negation_False normalized_context: {inst2.get('normalized_context', '')}")
            lines.append(
                f"  normalized answers: (True_run={info['norm_answer1']}, False_run={info['norm_answer2']})"
            )
            lines.append("")

    report_text = '\n'.join(lines)

    # to ensure output directory exists
    output_dir = f'./results/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    output_path = (
        f'./results/{dataset_name}/'
        f'DSR_eval_{today_date}_{today_time}_{dataset_name}_{model_name}_aristotle.txt'
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_files(args.dataset_name, args.model_name)
