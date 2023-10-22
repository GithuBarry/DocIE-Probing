import os
import torch, json
import numpy as np
from transformers import BertTokenizer, BertModel
import psutil
from tqdm import tqdm
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.to(device)

# Define the list of text to be embedded
dataset_name = "wikievents"
wikievent_input = open("../../Corpora/WikiEvents/gtt_format/all-test-dev-train.jsonl")
examples =[json.loads(l) for l in wikievent_input.readlines()]
wikievent_input.close()

text_list = [example['doctext'] for example in examples]

# Initialize an empty list to store the hidden states for each layer
hidden_states_list = [[] for _ in range(13)]
tokenized= []

# Loop over each text in the list and embed them individually
for text_idx, text in tqdm(enumerate(text_list)):
    # Truncate the text to 512 tokens
    truncated_text = text

    # Tokenize the truncated text
    tokenized_text = tokenizer.tokenize(truncated_text)[:512]
    tokenized.append(tokenized_text)

    # Convert the tokenized text to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Pad the input IDs to the maximum length of 512
    padded_input_ids = input_ids + [0]*(512-len(input_ids))

    # Convert the padded input IDs to a PyTorch tensor and move to the GPU
    input_ids_tensor = torch.tensor([padded_input_ids]).to(device)

    # Embed the input IDs using the BERT model
    outputs = model(input_ids_tensor)

    # Get the hidden states of the encoder
    hidden_states = outputs[2]

    # Append the hidden states for each layer to the corresponding list
    for i, layer in enumerate(hidden_states):
        hidden_states_list[i].append(layer.detach().cpu().numpy()[0])

    # Print the memory usage for the current loop iteration
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024
    tqdm.write(f'Memory usage for text {text_idx+1}: {memory_usage:.2f} MB')



# Convert each list of hidden states to a NumPy array
hidden_states_array = [np.array(layer_list) for layer_list in hidden_states_list]

# Save each hidden state array to disk
output_dir = 'hidden_states'
os.makedirs(output_dir, exist_ok=True)

for i, layer_array in enumerate(hidden_states_array):
    output_path = os.path.join(output_dir, f'X_raw_layer_{i}_bert-uncased.npy')
    np.save(output_path, layer_array)
    print(f'Saved hidden state array for layer {i} to {output_path}.')

json.dump(tokenized, open(f"tokenized_{dataset_name}.json", "w+"))