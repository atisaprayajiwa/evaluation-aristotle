import os
import re
import json
import threading
import argparse
import traceback
import shutil
import tempfile
import ast
import concurrent.futures
from tqdm import tqdm
from utils import OpenAIModel


class GPT3_True_Naive_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.batch_num = args.batch_num
        self.mode = args.mode
        self.file_lock = threading.Lock()

        if args.base_url:
            self.openai_api = OpenAIModel(
                args.api_key,
                args.model_name,
                args.stop_words,
                args.max_new_tokens,
                args.reasoning_effort,
                base_url=args.base_url
            )
        else:
            self.openai_api = OpenAIModel(
                args.api_key,
                args.model_name,
                args.stop_words,
                args.max_new_tokens,
                args.reasoning_effort
            )

    def load_true_naive_prompt(self):
        """
        Load the *true naive* prompt template.
        """
        file_path = os.path.join('./prompts', self.dataset_name, 'true_naive_prompt.txt')
        print("Loading true naive prompt file:", file_path)
        with open(file_path, encoding='utf-8') as f:
            template = f.read()
        return template

    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json'),
                  encoding='utf-8') as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def remove_think_blocks(self, text):
        """
        Strip out any <think>...</think> blocks if the model returns them.
        """
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def construct_true_naive_prompt(self, record, true_naive_template):
        """
        Construct the prompt by filling [[PREMISES]] and [[CONJECTURE]]
        in the true naive template.
        """
        if self.dataset_name == "LogicNLI":
            context = "\n".join(record['facts'] + record['rules'])
            conjecture = record['conjecture']
        else:
            context = record['context']
            # For non-LogicNLI datasets, we just use the question as the conjecture.
            q_text = record['question'].strip()
            conjecture = q_text

        full_prompt = true_naive_template.replace('[[PREMISES]]', context)
        full_prompt = full_prompt.replace('[[CONJECTURE]]', conjecture)
        return full_prompt, context, conjecture

    def parse_true_naive_output(self, response_text):
        """
        Parse the model output for the final label ONLY.
        The true_naive_prompt should instruct the model to produce exactly
        one of: TRUE, FALSE, UNKNOWN, SELF-CONTRADICTORY ...
        No explanation is expected or used.
        """
        text = response_text.strip()

        # Look for a direct match of the label (case-insensitive).
        mapping = {
            'TRUE': 'TRUE',
            'FALSE': 'FALSE',
            'SELF-CONTRADICTORY': 'SELF-CONTRADICTORY',
            'SELF_CONTRADICTORY': 'SELF-CONTRADICTORY',
            'CONTRADICTORY': 'SELF-CONTRADICTORY',
            'UNKNOWN': 'UNKNOWN'
        }

        candidates = re.findall(r'\b[A-Za-z][A-Za-z_\-]*\b', text)
        final_answer = ""

        # Search from the end to be robust to any leading text.
        for tok in reversed(candidates):
            up = tok.upper()
            if up in mapping:
                final_answer = mapping[up]
                break

        # As a fallback, try to match the full words directly in the text.
        if not final_answer:
            upper_text = text.upper()
            for key, val in mapping.items():
                if key in upper_text:
                    final_answer = val
                    break

        # If still nothing, just return the raw uppercased text (last resort),
        # but in most cases we expect one of the mapping keys.
        if not final_answer and text:
            final_answer = text.strip().upper()

        return final_answer

    def save_output(self, outputs):
        if "llama" in self.model_name:
            model_name = 'llama'
        elif "Qwen3-14B" in self.model_name:
            model_name = 'qwen3-14b'
        elif "Qwen3-32B" in self.model_name:
            model_name = 'qwen3-32b'
        elif "deepseek" in self.model_name:
            model_name = "deepseek"
        elif ":free" in self.model_name:
            model_name = self.model_name.replace(":free", "_free")
        else:
            model_name = self.model_name

        # New filename for true naive baseline (no explanation, no informative example in template)
        file_name = f'{self.dataset_name}_{model_name}_true_naive_prompt.json'
        file_path = os.path.join(self.save_path, self.dataset_name, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print("Saving result with thread lock in path:", file_path)
        with self.file_lock:
            try:
                existing_data = []
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        if file_content.strip():
                            try:
                                existing_data = json.loads(file_content)
                            except json.JSONDecodeError:
                                corrupt_backup = file_path + '.corrupt'
                                shutil.copy(file_path, corrupt_backup)
                                print(f"Warning: JSON decode error when reading {file_path}. "
                                      f"Backed up corrupted file to {corrupt_backup}.")

                                try:
                                    recovered = ast.literal_eval(file_content)
                                    if isinstance(recovered, list):
                                        existing_data = recovered
                                    else:
                                        existing_data = [recovered]
                                    print("Recovered file content via ast.literal_eval; continuing.")
                                except Exception:
                                    print("Could not recover corrupted file content; reinitializing output list.")
                                    existing_data = []
                else:
                    existing_data = []

                if isinstance(outputs, list):
                    existing_data.extend(outputs)
                else:
                    existing_data.append(outputs)

                dirpath = os.path.dirname(file_path) or '.'
                tmp_fd, tmp_path = tempfile.mkstemp(dir=dirpath)
                try:
                    with os.fdopen(tmp_fd, 'w', encoding='utf-8') as tmpf:
                        json.dump(existing_data, tmpf, indent=2, ensure_ascii=False)
                    os.replace(tmp_path, file_path)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

            except Exception as e:
                print(f"Error in saving output: {e}")

    def process_example(self, example, true_naive_template):
        prompt, context, conjecture = self.construct_true_naive_prompt(example, true_naive_template)

        print(f"[Example {example.get('id')}] Running TRUE naive prompt ...")
        # generate now returns (text, finish_reason, usage)
        resp_text, finish_reason, usage = self.openai_api.generate(prompt)

        # Use the raw text for downstream parsing
        resp = resp_text if isinstance(resp_text, str) else str(resp_text)

        print("Model response:", resp)
        think_removed_resp = self.remove_think_blocks(resp)
        final_answer = self.parse_true_naive_output(think_removed_resp)
        print("Parsed final answer:", final_answer)

        ground_truth = example.get('answer', example.get('label'))

        if self.dataset_name == 'LogicNLI':
            original_context = "Facts: " + '\n'.join(example['facts']) + "\nRules: " + '\n'.join(example['rules'])
            question = example['conjecture']
        else:
            original_context = example['context']
            question = example['question']
        
        output = {
            'id': example['id'],
            'original_context': original_context,
            'question': question,
            'conjecture': conjecture,
            'naive_prompt_output_raw': resp,
            'final_answer': final_answer,
            'ground_truth': ground_truth,
            'token_usage': usage or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        return output

    def run_true_naive_inference(self):
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        true_naive_template = self.load_true_naive_prompt()
        print("Number of batch workers:", self.batch_num)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_num) as executor:
            futures = {
                executor.submit(self.process_example, example, true_naive_template): example
                for example in raw_dataset
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(raw_dataset)):
                example = futures[future]
                try:
                    output = future.result()
                    if output is None:
                        print(f"No output for example {example['id']}")
                        continue

                    if not isinstance(output, dict):
                        print(f"Unexpected output type for {example['id']}: {type(output)} -- skipping save")
                        continue

                    print(f"Saving output for example: {output.get('id')}")
                    self.save_output(output)

                except Exception as exc:
                    print(f'{example["id"]} generated an exception: {exc}')
                    traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str, default='true_naive')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--base_url', type=str)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--reasoning_effort', type=str, default='none')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    runner = GPT3_True_Naive_Baseline(args)
    runner.run_true_naive_inference()