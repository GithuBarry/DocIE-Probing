import os
import torch
import json
import numpy as np
from transformers import BertTokenizer, BertModel
import psutil
from tqdm.auto import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate BERT embeddings for text documents')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input JSONL file')
    parser.add_argument('--output_dir', type=str, default='hidden_states',
                        help='Directory to save the hidden states (default: hidden_states)')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='BERT model to use (default: bert-base-uncased)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset (used for tokenized output filename)')
    return parser.parse_args()

def load_examples(input_file):
    with open(input_file) as f:
        examples = [json.loads(l) for l in f.readlines()]
    return examples

def generate_embeddings(args):
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name, output_hidden_states=True)
    model.to(device)

    # Load examples
    examples = load_examples(args.input_file)
    text_list = [example['doctext'] for example in examples]

    # Initialize lists for hidden states and tokenized text
    hidden_states_list = [[] for _ in range(13)]  # BERT has 12 layers + input embeddings
    tokenized = []

    # Process texts
    for text_idx, text in tqdm(enumerate(text_list), total=len(text_list)):
        # Tokenize and truncate
        tokenized_text = tokenizer.tokenize(text)[:args.max_length]
        tokenized.append(tokenized_text)

        # Convert tokens to IDs and pad
        input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        padded_input_ids = input_ids + [0] * (args.max_length - len(input_ids))

        # Convert to tensor and move to device
        input_ids_tensor = torch.tensor([padded_input_ids]).to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(input_ids_tensor)

        # Store hidden states
        hidden_states = outputs[2]
        for i, layer in enumerate(hidden_states):
            hidden_states_list[i].append(layer.detach().cpu().numpy()[0])

        # Log memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024
        if (text_idx + 1) % 100 == 0:  # Log every 100 items
            tqdm.write(f'Memory usage for text {text_idx+1}: {memory_usage:.2f} MB')

    # Convert to numpy arrays and save
    os.makedirs(args.output_dir, exist_ok=True)
    
    for i, layer_list in tqdm(enumerate(hidden_states_list), desc="Saving layers"):
        layer_array = np.array(layer_list)
        output_path = os.path.join(args.output_dir, f'X_raw_layer_{i}_{args.model_name}.npy')
        np.save(output_path, layer_array)
        print(f'Saved hidden state array for layer {i} to {output_path}')

    # Save tokenized text
    tokenized_output = os.path.join(args.output_dir, f"tokenized_{args.dataset_name}.json")
    with open(tokenized_output, "w") as f:
        json.dump(tokenized, f)
    print(f"Saved tokenized text to {tokenized_output}")

def main():
    args = parse_arguments()
    generate_embeddings(args)

if __name__ == "__main__":
    main()