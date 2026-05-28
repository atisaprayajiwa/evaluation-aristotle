import json
import os
from tqdm import tqdm
from utils import OpenAIModel
import argparse
import re
import sys
import concurrent.futures
import traceback
import threading
import shutil

class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
        self.batch_num = args.batch_num
        self.file_lock = threading.Lock()
        if args.base_url:
            self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens, args.reasoning_effort, base_url=args.base_url)
        else:
            self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens, args.reasoning_effort)
            
    def load_in_context_examples_trans(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'translation.txt')
        print("Loading translation file: ", file_path)
        with open(file_path, encoding='utf-8') as f:
            in_context_examples = f.read()
            
        return in_context_examples
    
    def load_in_context_and_or_decomposer(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'and_or_decomposer.txt')
        print("Loading decomposer file: ", file_path)
        with open(file_path, encoding='utf-8') as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_either_or_decomposer(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'either_or_decomposer.txt')
        print("Loading decomposer file: ", file_path)
        with open(file_path, encoding='utf-8') as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_biconditional_decomposer(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'logical_biconditional_decomposer.txt')
        print("Loading decomposer file: ", file_path)
        with open(file_path, encoding='utf-8') as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_examples_search_init(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'search_init.txt')
        with open(file_path, encoding='utf-8') as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_examples_search_router(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'search_router.txt')
        with open(file_path, encoding='utf-8') as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json'), encoding='utf-8') as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def remove_think_blocks(self, text):
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
   
    def index_context(self, context):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = '\n'.join([f"{index}: {sentence}" for index, sentence in formatted_context])
        return str(indexed_sentences)

    def construct_prompt_a(self, record, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        if self.dataset_name == "LogicNLI":
            context = "\n".join(record['facts'] + record['rules'])
            question = record['conjecture']
        else:
            context = record['context']
            question = re.search(r'\?(.*)', record['question'].strip()).group(1).strip()
        full_prompt = full_prompt.replace('[[PREMISES]]', context)
        full_prompt = full_prompt.replace('[[CONJECTURE]]', question)
        return full_prompt

    def construct_prompt_b(self, responses_a, in_context_examples_decomposer):
        full_prompt = in_context_examples_decomposer
        if isinstance(responses_a, list):
            responses_a = '\n'.join(responses_a)
        full_prompt = full_prompt.replace('[[PREMISES]]', responses_a)
        return full_prompt

    def construct_prompt_c(self, responses_b, in_context_examples_search_init):
        full_prompt = in_context_examples_search_init
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_b)
        return full_prompt
    
    def construct_prompt_d(self, responses_b, negated_label, reasoning_step, sos_list, in_context_examples_search_router):
        full_prompt = in_context_examples_search_router
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_b)
        full_prompt = full_prompt.replace('[[NEGATION-INITIALIZATION]]', negated_label)
        full_prompt = full_prompt.replace('[[REASONING]]', reasoning_step)
        full_prompt = full_prompt.replace('[[SOS]]', sos_list)
        return full_prompt
        
    def post_process_b(self, response_b):
        parts = response_b.split("Final Form:") 
        if len(parts) > 1:
            context_text = parts[-1].strip()
        else:
            context_text = "Context not found."
        
        return context_text
    
    def negate_conjecture(self, conjecture):
        if re.search(r'True', conjecture):
            updated_conjecture = re.sub(r'True', 'False', conjecture)
        elif re.search(r'False', conjecture):
            updated_conjecture = re.sub(r'False', 'True', conjecture)
        else:
            updated_conjecture = conjecture
            print("No True/False found in the conjecture.")

        return updated_conjecture
        

    def post_process_c(self, response_c):
        sos_list = re.findall(r'\[(.*?)\]', response_c)
        negated_label = re.findall(r'\{(.*?)\}', response_c)
        str_sos_list = "".join(sos_list)
        str_negated_label = "".join(negated_label)
        return str_negated_label.lower(), str_sos_list.lower()
    
    def post_process_d(self, response_d):
        search_result_match = re.findall(r'\{(.*?)\}', response_d)
        search_result = search_result_match[0].strip() if search_result_match else None
        
        new_clause_match = re.search(r"New Clause:\s*```(.*?)```", response_d, re.DOTALL)
        new_clause = new_clause_match.group(1).strip() if new_clause_match else None
        
        sufficiency_check_match = re.search(r"Sufficiency check for final answer:\s*(.*)", response_d)
        sufficiency_check = sufficiency_check_match.group(1).strip() if sufficiency_check_match else None
        
        sufficiency_label_match = re.search(r"Sufficiency Label:\s*\[(.*?)\]", response_d)
        sufficiency_label = sufficiency_label_match.group(1).strip() if sufficiency_label_match else None
        
        final_answer = "***Final Answer: N/A***"
        if sufficiency_label and sufficiency_label.lower() == "true":
            final_answer_match = re.search(r'\*\*\*(.*?)\*\*\*', response_d)
            final_answer = final_answer_match.group(1).strip() if final_answer_match else "No final answer found"
    
        
        return {
            "Search Result": search_result,
            "New Clause": new_clause,
            "Sufficiency Check": sufficiency_check,
            "Sufficiency Label": sufficiency_label,
            "Final Answer": final_answer
        }
        
    def process_normalized_context(self, normalized_context):
        lines = normalized_context.split('\n')
        
        for i, line in enumerate(lines):
            or_positions = [m.start() for m in re.finditer(r'\∨|\\lor', line)]
            for pos in or_positions:
                left_part = line[:pos]
                match = re.search(r'(True|False)(?!.*?(True|False).*?$)', left_part)
                if match:
                    if match.group() == 'True':
                        new_left_part = left_part[:match.start()] + 'False' + left_part[match.end():]
                        lines[i] = new_left_part + line[pos:]
        return '\n'.join(lines)
    
    def post_process_final_answer(self, response_c):
        pattern_bracket = r"Final answer: \{([A-E])\}"
        match = re.search(pattern_bracket, response_c)
        if match:
            answers =  match.group(1)
            return answers
        pattern_direct = r'\{(\w+)\}'
        match = re.search(pattern_direct, response_c, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "No final answer found in the text."

    
    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'A'
        elif final_answer == "false":
            final_answer = 'B'
        elif final_answer == "unknown":
            final_answer = 'C'
        else:
            final_answer = "No final answer found in the text."  
        return final_answer
    
    def list_to_indexed_string(self, item_list):
        indexed_list = [f"{i + 1}. {item}" for i, item in enumerate(item_list)]
        return "\n".join(indexed_list)
    
    def extract_facts_and_rules(self, content):
        fact_pattern = r'Facts(.*?)Rules'
        fact_match = re.search(fact_pattern, content, re.DOTALL)
        facts = fact_match.group(1).strip() if fact_match else None

        rules_pattern = r'Rules(.*?)Conjecture'
        rules_match = re.search(rules_pattern, content, re.DOTALL)
        rules = rules_match.group(1).strip() if rules_match else None

        return facts, rules

    def extract_query(self, content):
        query_matches = list(re.finditer(r'Conjecture', content))
        
        if not query_matches:
            return None
        
        last_query_pos = query_matches[-1].start()
        
        last_query_content = content[last_query_pos:]
        
        patterns = [
            r'Conjecture\s*```(?:plaintext)?\n?(.+?)\n?```',
            r'Conjecture\s*(.+?)(?:\n|$)',
            r'Conjecture.*?\n(.*?)(?:\n|$)',
            r'Conjecture.*?\n.*?\n(.*?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, last_query_content, re.DOTALL)
            if match:
                query = match.group(1).strip()
                if '(' in query and ')' in query:
                    return query
        
        return None
    
    def clean_irrelevant_lines(self, content):
        lines = content.split("\n")
        cleaned_lines = [line.strip() for line in lines if "(" in line]
        return "\n".join(cleaned_lines)
    
    def remove_duplicates(self, context_lines):
        seen = set()
        unique_items = []
        
        for item in context_lines:
            if item not in seen:
                unique_items.append(item)
                seen.add(item) 
        return unique_items
    
    def split_cnf_clause(self, context):
        print("Splitting context: ", context)
        normalized_context_lines = context.split('\n')
        unique_context_lines = self.remove_duplicates(normalized_context_lines)
        normalized_context = [line.split('\land') for line in unique_context_lines]
        flattened_normalized_context = [item for sublist in normalized_context for item in sublist]
        cleaned_normalized_context = [line.split(":::")[0].strip() for line in flattened_normalized_context]
        cleaned_normalized_context = [re.sub(r'^\d+\.\s*', '', line) for line in cleaned_normalized_context]
        renumbered_normalized_context = [f"{index + 1}. {line}" for index, line in enumerate(cleaned_normalized_context)]

        return renumbered_normalized_context
    
    def categorize_rule_lines(self, rule):
        either_or = []
        biconditional = []
        others = []

        for item in rule.split('\n'):
            if "(" in item:
                if '\u2295' in item:
                    either_or.append(item)
                elif '\u21d4' in item:
                    biconditional.append(item)
                else:
                    others.append(item)

        return either_or, biconditional, others
    
    def extract_facts_rules_conjecture(self, content):
        stop_markers = ["### Explanation", "Explanation:", "### Note", "### Output explanation"]
        for marker in stop_markers:
            if marker in content:
                content = content.split(marker)[0]
        
        pattern = r'Facts?[:\s]*(.*?)Rules?[:\s]*(.*?)Conjecture[:\s]*(.*)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            fact = match.group(1).strip()
            rule = match.group(2).strip()
            conjecture = match.group(3).strip()
        else:
            print("Warning: Standard pattern match failed, attempting fallback extraction.")
            fact_match = re.search(r'Facts?[:\s]*(.*?)Rules?', content, re.DOTALL | re.IGNORECASE)
            fact = fact_match.group(1).strip() if fact_match else ""
            rule_match = re.search(r'Rules?[:\s]*(.*?)Conjecture', content, re.DOTALL | re.IGNORECASE)
            rule = rule_match.group(1).strip() if rule_match else ""
            conj_match = re.search(r'Conjecture[:\s]*(.*)', content, re.DOTALL | re.IGNORECASE)
            conjecture = conj_match.group(1).strip() if conj_match else ""
            
        def clean_text(text):
            text = text.replace("```plaintext", "").replace("```", "")
            return text.strip()
        
        return clean_text(fact), clean_text(rule), clean_text(conjecture)


    def post_process_decompose(self, content):
        final_form_match = re.finditer(r'Final Form', content)
        final_form_positions = [m.start() for m in final_form_match]

        if final_form_positions:
            first_final_form_pos = final_form_positions[0]
            
            try:
                context = content[first_final_form_pos:].split("Final Form:")[-1].strip()

                context_lines = context.split("\n")
                filtered_context_lines = [line for line in context_lines if "(" in line]
                context = "\n".join(filtered_context_lines)
                
            except IndexError as e:
                problematic_content = content[first_final_form_pos:]
                print("Problematic content: ", problematic_content)
                context = ""
        else:
            context_lines = content.split("\n")
            filtered_context_lines = [line for line in context_lines if "(" in line]
            context = "\n".join(filtered_context_lines)

        return context
    
    def clean_conjecture(self, conjecture):
        if isinstance(conjecture, dict):
            conjecture = "\n".join([f"{key}: {value}" for key, value in conjecture.items()])
        splitted_conjecture = conjecture.replace('\\n', '\n').split('\n')
        cleaned_conjecture = []
        for item in splitted_conjecture:
            if "(" in item:
                remove_list = ['Rules (others):', 'Rules (biconditional):', 'Rules (either_or):']
                if not any(remove_item in item for remove_item in remove_list):
                    splited_item = item.split(":::")[0]
                    cleaned_conjecture.append(splited_item)
                
        return '\n'.join(cleaned_conjecture)
        
    def save_output(self, outputs, file_suffix=None):
        if "llama" in self.model_name:
            model_name = 'llama'
        elif "Qwen3-14B" in self.model_name:
            model_name = 'qwen3-14b'
        elif "Qwen3-32B" in self.model_name:
            model_name = 'qwen3-32b'
        elif "deepseek" in self.model_name:
            model_name = 'deepseek'
        else:
            model_name = self.model_name
        file_name = f'{self.dataset_name}_{model_name}_trans_decompose_no_negation.json'
        file_path = os.path.join(self.save_path, self.dataset_name, file_name)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        print("Saving result with thread lock in path: ", file_path)
        
        # Use a temporary file for atomic writing
        temp_file_path = file_path + ".tmp"
        
        with self.file_lock:
            try:
                existing_data = []
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            if file_content.strip():
                                existing_data = json.loads(file_content)
                    except json.JSONDecodeError as json_err:
                        print(f"CRITICAL ERROR: {file_path} is corrupted (JSONDecodeError).")
                        print(f"Details: {json_err}")
                        # Backup the corrupted file so we don't lose data
                        backup_path = file_path + ".corrupted"
                        print(f"Backing up corrupted file to {backup_path} and starting fresh.")
                        if os.path.exists(backup_path):
                            os.remove(backup_path) # overwrite old backup if exists
                        os.rename(file_path, backup_path)
                        existing_data = []

                if isinstance(outputs, list):
                    existing_data.extend(outputs)
                else:
                    existing_data.append(outputs)
                
                # Write to temp file first
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                # Atomically replace the old file with the new file
                os.replace(temp_file_path, file_path)
                
            except Exception as e:
                print(f"Error in saving output: {e}")
                traceback.print_exc()
                # Clean up temp file if it exists
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass

    def process_example(self, example, in_context_examples_trans, icl_and_or_decomposer, icl_either_or_decomposer, icl_biconditional_decomposer):
        # per-example token accounting 
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        def add_usage(u):
            if not u:
                return
            token_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            token_usage["completion_tokens"] += u.get("completion_tokens", 0)
            token_usage["total_tokens"] += u.get("total_tokens", 0)
        
        if self.dataset_name == 'LogicNLI':
            question = example['conjecture']
        else:
            question = example['question'].split('?')[1]

        print("Translating...")
        prompts_a = self.construct_prompt_a(example, in_context_examples_trans)
        trans_text, _, usage_trans = self.openai_api.generate(prompts_a)
        add_usage(usage_trans)
        responses_a = self.remove_think_blocks(trans_text)
        print("Translation response: ", responses_a)
        
        translated_facts, translated_rules, translated_conjecture = self.extract_facts_rules_conjecture(responses_a)
        print("Translated Facts1: ", translated_facts)
        translated_facts = self.clean_irrelevant_lines(translated_facts)
        print(f"Translated Facts2: {translated_facts}")

        either_or, biconditional, and_or = self.categorize_rule_lines(translated_rules)
        
        print(f"AND/OR: {and_or} \n EITHER/OR: {either_or} \n BICONDITIONAL: {biconditional}")

        responses_and_or = None
        responses_either_or = None
        responses_biconditional = None

        responses_and_or_process = None
        responses_either_or_process = None
        responses_biconditional_process = None

        print("Decomposing rules...")
        if and_or:
            prompt_and_or = self.construct_prompt_b(and_or, icl_and_or_decomposer)
            and_or_text, _, usage_and_or = self.openai_api.generate(prompt_and_or)
            add_usage(usage_and_or)

            responses_and_or_process = self.remove_think_blocks(and_or_text)
            responses_and_or = self.post_process_decompose(responses_and_or_process)
            responses_and_or = self.clean_irrelevant_lines(responses_and_or)

        if either_or:
            prompt_either_or = self.construct_prompt_b(either_or, icl_either_or_decomposer)
            either_or_text, _, usage_either_or = self.openai_api.generate(prompt_either_or)
            add_usage(usage_either_or)

            responses_either_or_process = self.remove_think_blocks(either_or_text)
            responses_either_or = self.post_process_decompose(responses_either_or_process)
            responses_either_or = self.clean_irrelevant_lines(responses_either_or)

        if biconditional:
            prompt_biconditional = self.construct_prompt_b(biconditional, icl_biconditional_decomposer)
            biconditional_text, _, usage_biconditional = self.openai_api.generate(prompt_biconditional)
            add_usage(usage_biconditional)

            responses_biconditional_process = self.remove_think_blocks(biconditional_text)
            responses_biconditional = self.post_process_decompose(responses_biconditional_process)
            responses_biconditional = self.clean_irrelevant_lines(responses_biconditional)
        
        responses_b = '\n'.join(filter(None, [responses_and_or, responses_either_or, responses_biconditional]))
        print("Decomposing response: ", responses_b)
        
        normalized_context = responses_b
        normalized_conjecture = self.clean_conjecture(translated_conjecture)
        print('Normalized context: ', normalized_context)
        print('Normalized conjecture: ', normalized_conjecture)
        
        negated_label = 'false'
        sos_list = normalized_conjecture

        print("Negated Label Initialization: ", negated_label)
        print("SOS List Initialization: ", sos_list)

        if self.dataset_name == "ProntoQA":
            normalized_context = self.process_normalized_context(normalized_context)

        if isinstance(normalized_context, list):
            normalized_context = "\n".join(self.split_cnf_clause(normalized_context))
            
        if self.dataset_name == 'LogicNLI':
            original_context = "Facts: " + '\n'.join(example['facts']) + "Rules: " + '\n'.join(example['rules'])
        else:
            original_context = example['context']

        output = {
            'id': example['id'], 
            'original_context': original_context,
            'question': question, 
            'raw_translation': responses_a,
            'translated_context': {"Translated_Facts": translated_facts, "Translated_Rules": translated_rules, "Translated_Conjecture": translated_conjecture},
            'decomposed_process': {key: value for key, value in {"and_or": locals().get("responses_and_or_process"), "either_or": locals().get("responses_either_or_process"), "biconditional": locals().get("responses_biconditional_process")}.items() if value is not None},
            'normalized_context': {"Fact": translated_facts, "and_or": responses_and_or, "either_or": responses_either_or, "biconditional": responses_biconditional},
            'normalized_conjecture': normalized_conjecture,
            'negated_label': negated_label,
            'sos_list': sos_list,
            'ground_truth': example['answer'],
            'token_usage': token_usage,
        }
        print("Token usage for the example", example['id'], ":", token_usage)
        print(output)
        return output, None
        
    def reasoning_graph_generation(self):
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        in_context_examples_trans = self.load_in_context_examples_trans()
        
        if self.dataset_name == "ProntoQA":
            icl_and_or_decomposer = self.load_in_context_and_or_decomposer()
            icl_either_or_decomposer = None
            icl_biconditional_decomposer = None
        else:
            icl_and_or_decomposer = self.load_in_context_and_or_decomposer()
            icl_either_or_decomposer = self.load_in_context_either_or_decomposer()
            icl_biconditional_decomposer = self.load_in_context_biconditional_decomposer()

        print("Number of batch: ", self.batch_num)
        counter = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_num) as executor:
            futures = {
                executor.submit(self.process_example, example, in_context_examples_trans, icl_and_or_decomposer, icl_either_or_decomposer, icl_biconditional_decomposer): example 
                for example in raw_dataset
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(raw_dataset)):
                example = futures[future]
                try:
                    output = future.result()
                    if 'error' in output:
                        print(f"Error in generating example: {example['id']}")
                    else:
                        print(f"Saving output for example: {output}")
                        self.save_output(output[0])
                except Exception as exc:
                    print(f'{example["id"]} generated an exception: {exc}')
                    traceback.print_exc()
                counter += 1
            
    
    def update_answer(self, sample, translation, decomposed_process, translated_fact, normalized_context, normalized_conjecture, negated_label, sos_list):
        if isinstance(normalized_context, list):
            normalized_context = "\n".join(normalized_context)
        
        output = {'id': sample['id'], 
                'original_context': sample['context'] if self.dataset_name != "LogicNLI" else "\n".join(sample['facts'] + sample['rules']),
                'question': sample['question'] if self.dataset_name != "LogicNLI" else sample['conjecture'], 
                'translated_context': translation,
                'decomposed_process': decomposed_process,
                'normalized_context': "Facts: " + translated_fact + "\nRules: " + normalized_context,
                'normalized_conjecture': normalized_conjecture,
                'negated_label': negated_label,
                'sos_list': sos_list,
                'ground_truth': sample['answer'] if self.dataset_name != "LogicNLI" else sample['label']}
        
        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    parser.add_argument('--base_url', type=str)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--reasoning_effort', type=str, default='none')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.reasoning_graph_generation()