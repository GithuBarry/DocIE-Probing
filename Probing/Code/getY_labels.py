import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Process MUC and WikiEvents datasets')
    parser.add_argument('--dataset', choices=['muc', 'wikievents'], required=True,
                       help='Dataset to process (muc or wikievents)')
    parser.add_argument('--input-file', required=True,
                       help='Path to input JSON/JSONL file')
    parser.add_argument('--output-dir', required=True,
                       help='Directory for output files')
    parser.add_argument('--type-roles-file', 
                       help='Path to EventTypeToRoles.json (required for wikievents)')
    parser.add_argument('--tokenized-files', nargs='+',
                       help='Paths to tokenized files with their model names in format path:model_name')
    return parser.parse_args()

def clean(string):
    """Clean text by removing non-alphanumeric characters and converting to lowercase."""
    string = string.lower()
    cleaned_string = re.sub(r'[^a-z0-9]', '', string)
    return cleaned_string

def find_substring_indices(string, substring):
    """Find all indices of substring in string."""
    indices = []
    length = len(substring)
    for i in range(len(string) - length + 1):
        if string[i:i + length] == substring:
            indices.append(i)
    return indices

def indices_in_tokenized_text(role_filler, tokens):
    """Find indices of role_filler in tokenized text."""
    cleaned_text = ""
    d = {}
    for i, token in enumerate(tokens):
        cleaned = clean(token)
        for ii in range(len(cleaned)):
            d[ii + len(cleaned_text)] = i
        cleaned_text += cleaned
    result = []
    for index in find_substring_indices(cleaned_text, clean(role_filler)):
        if index in d:
            result.append(d[index])
    return result

def generate_pairs(lists):
    """Generate all possible pairs from lists of items."""
    pairs = set()
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            for elem1 in lists[i]:
                for elem2 in lists[j]:
                    if elem1 != elem2:
                        l = sorted([elem1, elem2])
                        pairs.add((l[0], l[1]))
    return pairs

def n_comb(l):
    """Calculate number of possible combinations."""
    return len(l) * (len(l) - 1) // 2

def is_not_pronoun(phrase):
    """Check if phrase is not a pronoun using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    if phrase.lower().strip() in {
        "he", "him", "his", "himself",
        "she", "her", "hers",
        "it", "its",
        "they", "them", "their", "theirs", "themselves",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "one", "ones", "oneself",
        "who", "whom", "whose",
        "i", "me", "my", "mine", "myself", "that", "which", "some", "anyone",
        "those", "this", "these", "other", "another", "such", "each",
        "neither", "both", "all"
    }:
        return False
    
    doc = nlp(phrase)
    for token in doc:
        if token.text.lower() == "the":
            continue
        if token.pos_ == "PRON" or token.text.lower() in ["one", "ones"]:
            return False
    return True

def get_named_entities(text):
    """Extract named entities using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def load_dataset(file_path, dataset_type):
    """Load dataset from file with error handling."""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        with open(file_path) as f:
            if dataset_type == 'muc':
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("MUC dataset must be a JSON array")
                return data, 'muc1700'
            else:
                data = [json.loads(l) for l in f]
                if not data:
                    raise ValueError("WikiEvents dataset is empty")
                return data, 'wikievents246'
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def process_labels(examples, dataset_name):
    """Process and save label information."""
    label_sets = {
        "num_events": [len(example['templates']) for example in examples],
        "num_words": [len(nltk.word_tokenize(example['doctext'])) for example in examples],
        "num_sent": [len(nltk.sent_tokenize(example['doctext'])) for example in examples]
    }

    data = {}
    for key in label_sets:
        labels = label_sets[key]
        np.save(f"Y_{key}_{dataset_name}", np.array(labels))

        cuts = pd.qcut([-l for l in labels], q=10, duplicates='drop').categories
        labels_bucketed_10 = [len(cuts) - [-nt in cut for cut in cuts].index(True) - 1 for nt in labels]

        enc = OneHotEncoder()
        reshaped_array = np.array(labels_bucketed_10).reshape(-1, 1)
        labels_bucketed_10_onehot = enc.fit_transform(reshaped_array).toarray()
        np.save(f"Y_bucket_{key}_{dataset_name}", np.array(labels_bucketed_10_onehot))

        data[key] = dict(Counter(labels))
        data[key + "bucket"] = dict(Counter(labels_bucketed_10))

    return data

def validate_example(example, dataset_type):
    """Validate required fields in example."""
    required_fields = {'doctext', 'templates'}
    missing_fields = required_fields - set(example.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    if not isinstance(example['templates'], list):
        raise ValueError("'templates' must be a list")
    
    for template in example['templates']:
        if dataset_type == 'wikievents':
            if 'incident_type' not in template:
                raise ValueError("WikiEvents template missing 'incident_type'")
        else:  # muc
            if 'incident_type' not in template:
                raise ValueError("MUC template missing 'incident_type'")

def process_examples(examples, dataset_type, type_to_roles=None):
    """Process examples to extract various relationships with validation."""
    # Validate examples
    for i, example in enumerate(examples):
        try:
            validate_example(example, dataset_type)
        except ValueError as e:
            raise ValueError(f"Invalid example at index {i}: {str(e)}")
    counts = defaultdict(int)
    mention_count = []
    qualified_mention_count = []
    coref_pairs = []
    coref_negative_examples = []
    coref_pairs_non_single_mention = []
    coenv_positive_examples = []
    coenv_negative_examples = []
    event_typing_examples = []
    roling_examples = []
    all_role_examples = []
    
    # Set up event types based on dataset
    if dataset_type == 'muc':
        types = ["arson", "bombing", "kidnapping", "robbery", "forced work stoppage", "attack"]
    else:
        types = [t.split(".")[0] for t in type_to_roles.keys()]

    for ex in tqdm(examples):
        templates = ex['templates']
        text = ex['doctext']
        cleanedtext = clean(" ".join(text))
        
        # Initialize per-example containers
        template_of_entities = []
        flattened_list_all_templates = []
        per_example_coref_pairs = set()
        per_example_coref_pairs_non_single_mention = set()
        
        # Initialize lists for the current example
        coref_negative_examples.append(set())
        coenv_positive_examples.append(set())
        event_typing_examples.append(set())
        roling_examples.append(set())
        coenv_negative_examples.append(set())
        list_of_mention_per_template = []

        # Process each template
        for template in templates:
            flattened_list_within_template = []
            
            # Get arguments and roles based on dataset type
            if dataset_type == 'muc':
                arguments = [l for l in template.values() if isinstance(l, list)]
                roles = [v for v, l in template.items() if isinstance(l, list)]
            else:
                arguments = [l for l in template.values() if isinstance(l, list)]
                roles = type_to_roles[template['incident_type']]

            qualified_entities = []
            
            # Process each role and its fillers
            for role_index, role_fillers in enumerate(arguments):
                role = roles[role_index]
                list_of_entity_mentions = []
                
                for entity in role_fillers:
                    qualified_mentions = []
                    mentions_per_entity = []
                    
                    # Process mentions
                    mentions = entity if dataset_type == 'muc' else [m[0] for m in entity]
                    for mention in mentions:
                        if not is_not_pronoun(mention):
                            continue
                            
                        mentions_per_entity.append(mention)
                        roling_examples[-1].add((mention, role))
                        
                        if cleanedtext.count(clean(mention)) == 1:
                            qualified_mentions.append(mention)
                            counts['qualified_mentions'] += 1
                            
                            # Track duplicates
                            if mention in flattened_list_all_templates:
                                counts['duplicated qualified mention across templates'] += 1
                            flattened_list_all_templates.append(mention)
                            
                            if mention in flattened_list_within_template:
                                counts['duplicated qualified mention within templates'] += 1
                            flattened_list_within_template.append(mention)
                    
                    if qualified_mentions:
                        counts['qualified_entities'] += 1
                        qualified_entities.append(qualified_mentions)
                        per_example_coref_pairs.update(
                            [p for p in combinations(qualified_mentions, 2) if p[0] < p[1]]
                        )
                    
                    per_example_coref_pairs_non_single_mention.update(
                        [p for p in combinations(mentions_per_entity, 2) if p[0] < p[1]]
                    )
                    
                    mention_count.append(len(mentions_per_entity))
                    qualified_mention_count.append(len(qualified_mentions))
                    list_of_entity_mentions.append(mentions_per_entity)
                
                coref_negative_examples[-1].update(generate_pairs(list_of_entity_mentions))
            
            # Process event-level information
            counts['qualified_coenv'] += n_comb(set(flattened_list_within_template))
            coenv_positive_examples[-1].update(
                [p for p in combinations(flattened_list_within_template, 2) if p[0] < p[1]]
            )
            
            # Handle event typing based on dataset
            if dataset_type == 'muc':
                event_type = types.index(template["incident_type"].split("/")[0].strip().lower())
            else:
                event_type = types.index(template["incident_type"].split(".")[0])
                
            event_typing_examples[-1].update(
                [p + (event_type,) for p in combinations(flattened_list_within_template, 2) if p[0] < p[1]]
            )
            
            list_of_mention_per_template.append(flattened_list_within_template)
            template_of_entities.append(qualified_entities)
            
        # Process example-level information
        all_role_examples.append(flattened_list_all_templates)
        coenv_negative_examples[-1].update(generate_pairs(list_of_mention_per_template))
        counts['actual_coref_pairs'] += len(per_example_coref_pairs)
        coref_pairs.append(per_example_coref_pairs)
        counts['actual_coref_pairs_non_single'] += len(per_example_coref_pairs_non_single_mention)
        coref_pairs_non_single_mention.append(per_example_coref_pairs_non_single_mention)

    return {
        'counts': counts,
        'coref_pairs': coref_pairs,
        'coref_negative_examples': coref_negative_examples,
        'coenv_positive_examples': coenv_positive_examples,
        'coenv_negative_examples': coenv_negative_examples,
        'event_typing_examples': event_typing_examples,
        'roling_examples': roling_examples,
        'all_role_examples': all_role_examples,
        'coref_pairs_non_single_mention': coref_pairs_non_single_mention
    }

def process_named_entities(examples):
    """Process named entities using multiprocessing."""
    with Pool(cpu_count()) as pool:
        return list(tqdm(
            pool.imap(get_named_entities, [ex['doctext'] for ex in examples]),
            total=len(examples)
        ))

def generate_data(processed_results, tokenized_data, dataset_splits):
    """Generate final dataset with different model tokenizations."""
    result_data = {
        'coref': defaultdict(lambda: defaultdict(list)),
        'coenv': defaultdict(lambda: defaultdict(list)),
        'typing': defaultdict(lambda: defaultdict(list)),
        'rolelabeling': defaultdict(lambda: defaultdict(list)),
        'isarg': defaultdict(lambda: defaultdict(list))
    }
    
    for tokens, model_name in tokenized_data:
        # Process each task
        for task, task_data in [
            ('coref', (processed_results['coref_pairs_non_single_mention'], processed_results['coref_negative_examples'])),
            ('coenv', (processed_results['coenv_positive_examples'], processed_results['coenv_negative_examples'])),
            ('typing', processed_results['event_typing_examples']),
            ('rolelabeling', processed_results['roling_examples']),
            ('isarg', processed_results['all_role_examples'])
        ]:
            all_results = set()
            
            if task in ['coref', 'coenv']:
                pos_examples, neg_examples = task_data
                for label, examples in [(1, pos_examples), (0, neg_examples)]:
                    for doc_i, pairs in enumerate(examples):
                        for pair in pairs:
                            # Process pair data
                            indices0 = indices_in_tokenized_text(pair[0], tokens[doc_i])
                            indices1 = indices_in_tokenized_text(pair[1], tokens[doc_i])
                            
                            # Generate results
                            for idx0 in indices0:
                                for idx1 in indices1:
                                    result = {"doc_i": doc_i, "index0": idx0, "index1": idx1, "label": label}
                                    if tuple(result.values()) not in all_results:
                                        all_results.add(tuple(result.values()))
                                        split = dataset_splits(doc_i)
                                        result_data[task][model_name][split].append(result)
            
            elif task == 'typing':
                for doc_i, pairs in enumerate(task_data):
                    for pair in pairs:
                        label = pair[2]
                        indices0 = indices_in_tokenized_text(pair[0], tokens[doc_i])
                        indices1 = indices_in_tokenized_text(pair[1], tokens[doc_i])
                        
                        for idx0 in indices0:
                            for idx1 in indices1:
                                result = {"doc_i": doc_i, "index0": idx0, "index1": idx1, "label": label}
                                if tuple(result.values()) not in all_results:
                                    all_results.add(tuple(result.values()))
                                    split = dataset_splits(doc_i)
                                    result_data[task][model_name][split].append(result)
            
            elif task == 'rolelabeling':
                for doc_i, pairs in enumerate(task_data):
                    for pair in pairs:
                        mention, label = pair
                        indices = indices_in_tokenized_text(mention, tokens[doc_i])
                        
                        for idx in indices:
                            result = {"doc_i": doc_i, "index0": idx, "label": label}
                            if tuple(result.values()) not in all_results:
                                all_results.add(tuple(result.values()))
                                split = dataset_splits(doc_i)
                                result_data[task][model_name][split].append(result)
            
            elif task == 'isarg':
                ne_pos_examples = [list(set(ex)) for ex in task_data]
                ne_examples = [[p[0] for p in ex if p[1] in {"PERSON", "ORG"}] 
                             for ex in processed_results['named_entities']]
                ne_neg_examples = [[n for n in ne_examples[i] 
                                  if n not in " ".join(ne_pos_examples[i])] 
                                 for i in range(len(ne_examples))]
                
                for examples, label in [(ne_pos_examples, 1), (ne_neg_examples, 0)]:
                    for doc_i, mentions in enumerate(examples):
                        for mention in mentions:
                            indices = indices_in_tokenized_text(mention, tokens[doc_i])
                            
                            for idx in indices:
                                result = {"doc_i": doc_i, "index0": idx, "label": label}
                                if tuple(result.values()) not in all_results:
                                    all_results.add(tuple(result.values()))
                                    split = dataset_splits(doc_i)
                                    result_data[task][model_name][split].append(result)
    
    return result_data

from typing import Dict, List, Tuple, Callable, Any, Optional, Set, Union

class DataProcessor:
    """Main class for processing dataset and generating outputs.
    
    Attributes:
        args: Command line arguments
        output_dir: Path to output directory
        
    Methods:
        process(): Main processing pipeline
        get_dataset_splits(): Returns split determination function
        load_tokenized_files(): Loads and validates tokenized files
        align_tokenizations(): Ensures consistency across tokenizations
    """
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_dataset_splits(self):
        """Returns a function that determines dataset split based on index."""
        if self.args.dataset == 'muc':
            def split_func(idx):
                if idx < 200:
                    return "test"
                elif idx < 400:
                    return "dev"
                return "train"
        else:  # wikievents
            def split_func(idx):
                if idx < 20:
                    return "test"
                elif idx < 40:
                    return "dev"
                return "train"
        return split_func
    
    def load_tokenized_files(self):
        """Load and process tokenized files with validation."""
        if not self.args.tokenized_files:
            return []
        
        tokenized_data = []
        for file_spec in self.args.tokenized_files:
            try:
                if ':' not in file_spec:
                    raise ValueError(
                        f"Invalid tokenized file specification: {file_spec}. "
                        "Format should be 'path:model_name'"
                    )
                    
                path, model_name = file_spec.split(':')
                if not Path(path).exists():
                    raise FileNotFoundError(f"Tokenized file not found: {path}")
                
                with open(path) as f:
                    tokens = json.load(f)
                    if not isinstance(tokens, (list, dict)):
                        raise ValueError(
                            f"Invalid tokens format in {path}. "
                            "Expected list or dict of tokens."
                        )
                    tokenized_data.append((tokens, model_name))
                    
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {path}: {str(e)}")
            except Exception as e:
                raise Exception(f"Error loading tokenized file {path}: {str(e)}")
                
        return tokenized_data
    
    def align_tokenizations(self, tokenized_data, examples):
        """Align different model tokenizations to ensure consistency."""
        if not tokenized_data:
            return tokenized_data
            
        reference_tokens = tokenized_data[0][0]  # Use first tokenization as reference
        for doc_id in tqdm(range(len(examples)), desc="Aligning tokenizations"):
            ref_clean = clean(" ".join(reference_tokens[doc_id]))
            
            for i, (tokens, _) in enumerate(tokenized_data[1:], 1):
                truncate = 0
                total_len = 0
                while total_len < len(ref_clean) and truncate < len(tokens[doc_id]):
                    total_len += len(clean(tokens[doc_id][truncate]))
                    truncate += 1
                tokenized_data[i][0][doc_id] = tokens[doc_id][:truncate]
        
        return tokenized_data
    
    def process(self):
        """Main processing pipeline with progress tracking."""
        print("Starting data processing pipeline...")
        
        print("Loading dataset...")
        examples, dataset_name = load_dataset(self.args.input_file, self.args.dataset)
        print(f"Loaded {len(examples)} examples from {dataset_name}")
        
        print("Processing labels...")
        label_data = process_labels(examples, dataset_name)
        self.output_dir.joinpath(f'label_stats_{dataset_name}.json').write_text(
            json.dumps(label_data, indent=2)
        )
        
        if self.args.dataset == 'wikievents':
            print("Loading type_to_roles for WikiEvents...")
            if not self.args.type_roles_file:
                raise ValueError("type_roles_file is required for wikievents dataset")
            with open(self.args.type_roles_file) as f:
                type_to_roles = json.load(f)
        else:
            type_to_roles = None
        
        print("Processing examples...")
        processed_results = process_examples(examples, self.args.dataset, type_to_roles)
        
        print("Processing named entities...")
        processed_results['named_entities'] = process_named_entities(examples)
        
        print("Loading and aligning tokenized files...")
        tokenized_data = self.load_tokenized_files()
        tokenized_data = self.align_tokenizations(tokenized_data, examples)
        
        print("Generating final data...")
        split_func = self.get_dataset_splits()
        result_data = generate_data(processed_results, tokenized_data, split_func)
        
        print("Saving results...")
        for task, task_data in result_data.items():
            output_file = self.output_dir.joinpath(f'XY_{task}.json')
            output_file.write_text(json.dumps(task_data, indent=2))
            
        print("Processing completed successfully!")
        # Load dataset
        examples, dataset_name = load_dataset(self.args.input_file, self.args.dataset)
        
        # Process labels
        label_data = process_labels(examples, dataset_name)
        self.output_dir.joinpath(f'label_stats_{dataset_name}.json').write_text(
            json.dumps(label_data, indent=2)
        )
        
        # Load type_to_roles for wikievents
        type_to_roles = None
        if self.args.dataset == 'wikievents':
            if not self.args.type_roles_file:
                raise ValueError("type_roles_file is required for wikievents dataset")
            with open(self.args.type_roles_file) as f:
                type_to_roles = json.load(f)
        
        # Process examples
        processed_results = process_examples(examples, self.args.dataset, type_to_roles)
        
        # Add named entities
        processed_results['named_entities'] = process_named_entities(examples)
        
        # Load and align tokenized files
        tokenized_data = self.load_tokenized_files()
        tokenized_data = self.align_tokenizations(tokenized_data, examples)
        
        # Generate final data
        split_func = self.get_dataset_splits()
        result_data = generate_data(processed_results, tokenized_data, split_func)
        
        # Save results
        for task, task_data in result_data.items():
            output_file = self.output_dir.joinpath(f'XY_{task}.json')
            output_file.write_text(json.dumps(task_data, indent=2))

def main():
    args = parse_args()
    processor = DataProcessor(args)
    processor.process()

if __name__ == "__main__":
    main()