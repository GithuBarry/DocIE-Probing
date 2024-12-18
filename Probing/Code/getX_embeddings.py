from collections import defaultdict
import json
import os
import argparse
from tqdm import tqdm
import numpy as np

def extract_dygie_embeddings(args):
    """Extract embeddings from DyGIE++ model outputs"""
    examples_layers = defaultdict(list)
    
    with open(args.input_file) as f:
        muc_ids = [example['docid'] for example in json.load(f)]

    for muc_id in tqdm(muc_ids, desc="Processing DyGIE++ files"):
        npy = np.load(os.path.join(args.hidden_state_path, muc_id + ".npy"))
        for i in range(len(npy)):
            examples_layers[i].append(npy[i] if len(npy[i]) > 1 else npy[i][0])
    
    return examples_layers

def extract_gtt_embeddings(args):
    """Extract embeddings from GTT model outputs"""
    examples_layers = defaultdict(list)
    
    for split in ["test", "dev", "train"]:
        folder = split + "_hidden_states"
        folder_path = os.path.join(args.hidden_state_path, folder)
        
        if not os.path.exists(folder_path):
            continue
            
        for example in tqdm(os.listdir(folder_path), desc=f"Processing GTT {split} files"):
            npy = np.load(os.path.join(folder_path, example))
            for i in range(len(npy)):
                examples_layers[i].append(npy[i] if len(npy[i]) > 1 else npy[i][0])
    
    return examples_layers

def extract_tanl_embeddings(args):
    """Extract embeddings from TANL model outputs"""
    examples_layers = defaultdict(list)
    
    with open(args.input_file) as f:
        muc_ids = [example['docid'] for example in json.load(f)]

    for i, _ in enumerate(tqdm(muc_ids, desc="Processing TANL files")):
        file_name = f"output_sentence{i}_encoder_hidden_states.npy"
        npy = np.load(os.path.join(args.hidden_state_path, file_name))
        for j in range(len(npy)):
            examples_layers[j].append(npy[j] if len(npy[j]) > 1 else npy[j][0])
    
    return examples_layers


def save_embeddings(examples_layers, args):
    """Save extracted embeddings to files"""
    for i, layer_key in enumerate(tqdm(examples_layers.keys(), desc="Saving embeddings")):
        layer_name = f"layer_{layer_key}" if i < len(examples_layers.keys()) - 1 else "layer_last"
        file_name = os.path.join(args.output_dir, f"X_{args.model}_{layer_name}_{args.meta_data}")
        
        try:
            nparray = np.array(examples_layers[layer_key])
            np.save(file_name, nparray)
        except:
            print(f"Cannot save as npy because of varying embedding length, saving as npz: {file_name}")
            np.savez(file_name, examples_layers[layer_key])

def main():
    parser = argparse.ArgumentParser(description="Unified embeddings extraction script")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, 
                        choices=["dygiepp", "GTT", "TANL"],
                        help="Model type to process embeddings for")
    parser.add_argument("--hidden_state_path", type=str, required=True,
                        help="Path to the hidden states directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save output embeddings")
    
    # Optional arguments
    parser.add_argument("--muc_file", type=str,
                        help="Data file path (required for dygiepp, and TANL). Expect a list of document each having key `docid`")
    parser.add_argument("--meta_data", type=str, default="bert-uncased_epoch20",
                        help="Metadata string to include in output filenames")

    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate arguments
    if args.model in ["dygiepp", "TANL"] and not args.input_file:
        parser.error(f"{args.model} requires --input_file")
    
    # Process embeddings based on model type
    if args.model == "dygiepp":
        examples_layers = extract_dygie_embeddings(args)
        save_embeddings(examples_layers, args)
    
    elif args.model == "GTT":
        examples_layers = extract_gtt_embeddings(args)
        save_embeddings(examples_layers, args)
    
    elif args.model == "TANL":
        examples_layers = extract_tanl_embeddings(args)
        save_embeddings(examples_layers, args)


if __name__ == "__main__":
    main()