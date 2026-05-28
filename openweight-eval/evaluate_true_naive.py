import json
import argparse
import os
from datetime import datetime

LABELS = ['A', 'B', 'C', 'D']


def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_token_stats(all_items):
    stats = {
        'prompt_tokens': [],
        'completion_tokens': [],
        'total_tokens': []
    }
    # collect values + id
    for item in all_items:
        id_ = item.get('id', 'UNKNOWN_ID')
        usage = item.get('token_usage') or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        stats['prompt_tokens'].append((usage["prompt_tokens"], id_))
        stats['completion_tokens'].append((usage["completion_tokens"], id_))
        stats['total_tokens'].append((usage["total_tokens"], id_))

    token_stat_out = {}
    for key in stats:
        arr = stats[key]
        values = [v for v, _ in arr]
        if len(values) == 0:
            token_stat_out[key] = None
            continue
        min_val, min_id = min(arr, key=lambda x: x[0])
        max_val, max_id = max(arr, key=lambda x: x[0])
        avg_val = sum(values) / len(values)
        token_stat_out[key] = {
            "min": min_val,
            "min_id": min_id,
            "max": max_val,
            "max_id": max_id,
            "avg": avg_val
        }
    return token_stat_out

def map_final_answer_to_choice(final_answer_raw):
    if final_answer_raw is None:
        return 'C'

    s = str(final_answer_raw).strip().upper()

    if s in ['TRUE', 'T', 'A']:
        return 'A'

    if s in ['FALSE', 'F', 'NO', 'B']:
        return 'B'

    if s in ['UNKNOWN', 'C', 'NOT ENOUGH INFORMATION', 'UNKNOWN', 'NOT ENOUGH INFORMATION']:
        return 'C'

    if s in ['SELF-CONTRADICTORY', 'CONTRADICTORY', 'SELF_CONTRADICTORY', 'D']:
        return 'D'

    # this is the default where we can assume unknown
    return 'C'


def evaluate_instance(item):
    """
    Evaluate a single instance:
    - Read ground_truth ('A'/'B'/'C'/'D')
    - Map final_answer text to a choice
    - Return (is_correct, predicted_choice)
    """
    ground_truth = item.get('ground_truth', None)
    final_answer_raw = item.get('final_answer', None)

    if ground_truth is None:
        return False, None

    pred_choice = map_final_answer_to_choice(final_answer_raw)
    is_correct = (pred_choice == ground_truth)
    return is_correct, pred_choice


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
    # rows: ground truth, columns: prediction
    header = [' '] + LABELS
    lines = ['\t'.join(header)]
    for gt in LABELS:
        row = [gt] + [str(confusion[gt][pred]) for pred in LABELS]
        lines.append('\t'.join(row))
    return '\n'.join(lines)


def evaluate_file(dataset_name, model_name):

    file_path = f'./results/{dataset_name}/{dataset_name}_{model_name}_true_naive_prompt.json'
    data = load_json_file(file_path)

    confusion = init_confusion_matrix()

    total_instances = 0
    correct_instances = 0
    error_ids = []
    correct_ids = []

    error_examples = {}

    for item in data:
        id_ = item.get('id', 'UNKNOWN_ID')
        ground_truth = item.get('ground_truth', None)

        if not ground_truth:
            continue

        is_correct, pred_choice = evaluate_instance(item)

        # to update  confusion matrix (only if both labels are valid)
        if ground_truth in LABELS and pred_choice in LABELS:
            confusion[ground_truth][pred_choice] += 1

        total_instances += 1

        if is_correct:
            correct_instances += 1
            correct_ids.append(id_)
        else:
            error_ids.append(id_)
            key = (ground_truth, pred_choice)
            # to save the first example for any of the non-diagonal cells
            if (
                ground_truth in LABELS
                and pred_choice in LABELS
                and ground_truth != pred_choice
                and key not in error_examples
            ):
                error_examples[key] = item

    accuracy = correct_instances / total_instances if total_instances > 0 else 0.0
    
    token_stats = compute_token_stats(data)
    
    # to compute precision or recall and other metrics
    metrics = compute_metrics(confusion)

    # to print basic results
    print(f"Dataset name: {dataset_name}")
    print(f"Model name: {model_name}")
    print(f"Total instances (with ground truth): {total_instances}")
    print("\nToken Usage Statistics:")
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        s = token_stats[key]
        print(f"\n{key}:")
        print(f"  min: {s['min']} (id={s['min_id']})")
        print(f"  max: {s['max']} (id={s['max_id']})")
        print(f"  avg: {s['avg']:.2f}")
    print(f"Accuracy: {accuracy:.2%}")
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

    # to print example errors per non-zero wrong cell
    if error_examples:
        print("\nExample errors per (ground_truth, prediction) cell:")
        for (gt, pred), item in sorted(error_examples.items()):
            print(f"\nGround truth: {gt}, Predicted: {pred}")
            print(f"  id: {item.get('id', '')}")
            print(f"  original_context: {item.get('original_context', '')}")
            print(f"  question: {item.get('question', '')}")
            print(f"  conjecture: {item.get('conjecture', '')}")
            print(f"  naive_prompt_output_raw: {item.get('naive_prompt_output_raw', '')}")
            print(f"  ground_truth: {item.get('ground_truth', '')}")
            print(f"  predicted_choice: {map_final_answer_to_choice(item.get('final_answer', None))}")

    # to prepare report content for dumping to file
    now = datetime.now()
    today_date = now.strftime('%Y%m%d')
    today_time = now.strftime('%H%M%S')

    lines = []
    lines.append(f"Dataset name: {dataset_name}")
    lines.append(f"Model name: {model_name}")
    lines.append(f"Date: {today_date}")
    lines.append(f"Time: {today_time}")
    lines.append("")
    lines.append(f"Total instances (with ground truth): {total_instances}")
    lines.append(f"Accuracy: {accuracy:.4f}")
    lines.append("")
    lines.append("Token Usage Statistics:")
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        s = token_stats[key]
        lines.append(f"{key}:")
        lines.append(f"  min: {s['min']} (id={s['min_id']})")
        lines.append(f"  max: {s['max']} (id={s['max_id']})")
        lines.append(f"  avg: {s['avg']:.2f}")
        lines.append("")
        
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
        for (gt, pred), item in sorted(error_examples.items()):
            lines.append(f"Ground truth: {gt}, Predicted: {pred}")
            lines.append(f"  id: {item.get('id', '')}")
            lines.append(f"  original_context: {item.get('original_context', '')}")
            lines.append(f"  question: {item.get('question', '')}")
            lines.append(f"  conjecture: {item.get('conjecture', '')}")
            lines.append(f"  naive_prompt_output_raw: {item.get('naive_prompt_output_raw', '')}")
            lines.append(f"  ground_truth: {item.get('ground_truth', '')}")
            lines.append(f"  predicted_choice: {map_final_answer_to_choice(item.get('final_answer', None))}")
            lines.append("")

    report_text = '\n'.join(lines)

    # to ensure output directory does exist
    output_dir = f'./results/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    output_path = (
        f'./results/{dataset_name}/'
        f'thenondsreval_{today_date}_{today_time}_{dataset_name}_{model_name}_true_naive_prompt.txt'
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
    evaluate_file(args.dataset_name, args.model_name)