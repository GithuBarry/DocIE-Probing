import argparse
import gc
import json
import os
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from senteval_classifier import PyTorchClassifier, get_optimizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WeightedEmbeddingSumLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int = 1):
        super(WeightedEmbeddingSumLayer, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.linears = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_heads)])
        self.num_heads = num_heads

    def forward(self, inputs):
        # inputs shape: (batch_size, sequence_length, hidden_size)
        weighted_sums = []
        for linear in self.linears:
            # Apply linear layer to get attention weights
            weights = linear(inputs)
            # weights shape: (batch_size, sequence_length, 1)

            # Apply softmax activation to get attention probabilities
            attention_probs = self.softmax(weights)
            # attention_probs shape: (batch_size, sequence_length, 1)

            # Compute the weighted sum of embeddings
            weighted_sums.append(torch.sum(inputs * attention_probs, dim=1))
            # weighted_sum shape: (batch_size, hidden_size)

        if self.num_heads > 1:
            # Concatenate all weighted sums along the last dimension
            return torch.cat(weighted_sums, dim=-1)
        else:
            return weighted_sums[0]


class MLP(PyTorchClassifier):
    def __init__(self, params: Dict, inputdim: int, nclasses: int, num_heads: int = 1,
                 l2reg: float = 0., batch_size: int = 64, seed: int = 1111,
                 cudaEfficient: bool = False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)

        self.nhid = params.get("nhid", 0)
        self.optim = params.get("optim", "adam")
        self.tenacity = params.get("tenacity", 5)
        self.epoch_size = params.get("epoch_size", 4)
        self.max_epoch = params.get("max_epoch", 200)
        self.dropout = params.get("dropout", 0.)
        self.batch_size = params.get("batch_size", 64)
        self.num_heads = num_heads

        if self.nhid == 0:
            print("Regression!")
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            ).to(device)
        else:
            effective_input_dim = self.inputdim * num_heads if num_heads > 1 else self.inputdim
            self.model = nn.Sequential(
                WeightedEmbeddingSumLayer(self.inputdim, num_heads),
                nn.Linear(effective_input_dim, self.nhid),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(self.nhid, self.nclasses),
            ).to(device)

        self.loss_fn = nn.MSELoss().to(device)
        self.loss_fn.size_average = False

        optim_fn, optim_params = get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg


class ProbingTask:
    def __init__(self, args):
        self.args = args
        self.setup_environment()

    def setup_environment(self, seed=11):
        """Set up random seeds and CUDA"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_and_process_data(self, x_path: str, y_path: str) -> Tuple[np.ndarray, Union[np.ndarray, Dict]]:
        """Load X and Y data from paths"""
        X = np.load(x_path, allow_pickle=True)

        if y_path.endswith('.json'):
            Y = json.load(open(y_path))
        else:
            Y = np.load(y_path)

        return X, Y

    def reshape_x(self, X: np.ndarray, x_name: str) -> np.ndarray:
        """Reshape X data based on specific conditions"""
        if ("dygie" in x_name.lower() or "sent" in x_name.lower() or
                (not hasattr(X, "shape")) or len(X.shape) < 2):

            pickled_X = X
            new_X = []
            while len(pickled_X) == 1:
                if isinstance(pickled_X, np.ndarray):
                    pickled_X = pickled_X[0]
                else:
                    pickled_X = pickled_X["arr_0"]

            for example_X in pickled_X:
                while len(example_X) == 1:
                    example_X = example_X[0]
                new_X.append(example_X)

            X = new_X
            max_length = max([len(x) for x in X]) if not self.args.truncate else self.args.truncate_len

            for i in range(len(X)):
                if len(X[i]) < max_length:
                    X[i] = np.pad(X[i], ((0, max_length - len(X[i])), (0, 0)),
                                  'constant', constant_values=(0))
                if len(X[i]) > max_length:
                    X[i] = X[i][:max_length]

            X = np.array(X)
        return X

    def process_annotations(self, X: np.ndarray, annotation: Dict, x_name: str) -> Tuple[Dict, Dict]:
        """Process annotations for coref/coenv tasks"""
        for model_name in sorted(list(annotation.keys()), key=len, reverse=True):
            if model_name in x_name.lower():
                annotation = annotation[model_name]
                break

        Y = {}
        X_annotated = {}

        label_set = set()
        for partition in annotation:
            for example in annotation[partition]:
                label_set.add(example['label'])

        label_list = sorted(list(label_set))

        for partition in annotation.keys():
            for example in tqdm(annotation[partition], desc=partition):
                l = []
                idx = 0
                while f"index{idx}" in example:
                    l.append(X[example['doc_i']][example[f"index{idx}"]])
                    idx += 1

                idx = label_list.index(example['label'])
                label = [[1 if idx == i else 0 for i in range(len(label_list))]]

                if partition not in X_annotated:
                    X_annotated[partition] = np.array([l])
                    Y[partition] = np.array(label)
                else:
                    X_annotated[partition] = np.concatenate((X_annotated[partition], np.array([l])))
                    Y[partition] = np.concatenate((Y[partition], label))

        return X_annotated, Y

    def split_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/val/test sets"""
        if self.args.task_type == 'universal_labels':
            return (X[self.args.test_len + self.args.dev_len:],
                    X[self.args.test_len:self.args.test_len + self.args.dev_len],
                    X[:self.args.test_len],
                    Y[self.args.test_len + self.args.dev_len:],
                    Y[self.args.test_len:self.args.test_len + self.args.dev_len],
                    Y[:self.args.test_len])
        else:
            return (X['train'], X['dev'], X['test'],
                    Y['train'], Y['dev'], Y['test'])

    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test,
                           embedding_dimension, output_dimension) -> Dict:
        """Train model and evaluate performance"""
        mlp_classifier = MLP(
            params={"max_epoch": self.args.max_epoch,
                    "nhid": self.args.nhid,
                    "optim": "adam",
                    "tenacity": 10,
                    "batch_size": 8,
                    "dropout": 0.0},
            inputdim=embedding_dimension,
            nclasses=output_dimension,
            num_heads=self.args.num_heads,
            cudaEfficient=not torch.cuda.is_available()
        )

        print("Fitting MLP Classifier")
        mlp_classifier.fit(X_train, y_train, (X_val, y_val), early_stop=True)

        # Evaluate
        x_torch, y_torch = torch.from_numpy(X_train).to(device, dtype=torch.float32), torch.from_numpy(y_train).to(
            device, dtype=torch.int64)
        train_acc, p_train = mlp_classifier.score_and_prob(x_torch, y_torch)

        x_torch, y_torch = torch.from_numpy(X_val).to(device, dtype=torch.float32), torch.from_numpy(y_val).to(device,
                                                                                                               dtype=torch.int64)
        val_acc, p_val = mlp_classifier.score_and_prob(x_torch, y_torch)

        x_torch, y_torch = torch.from_numpy(X_test).to(device, dtype=torch.float32), torch.from_numpy(y_test).to(device,
                                                                                                                 dtype=torch.int64)
        test_acc, p_test = mlp_classifier.score_and_prob(x_torch, y_torch)

        y_true = [labels for _, labels in [(p_test[i], y_test[i]) for i in range(len(X_test))]]
        y_val_true = [labels for _, labels in [(p_val[i], y_val[i]) for i in range(len(X_val))]]

        return {
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "hidden_size": mlp_classifier.nhid,
            "inputdim/embeddim": mlp_classifier.inputdim,
            "output_dimension": output_dimension,
            "nclasses": mlp_classifier.nclasses,
            "batch_size": mlp_classifier.batch_size,
            "dropout": mlp_classifier.dropout,
            "l2reg": mlp_classifier.l2reg,
            "optim": mlp_classifier.optim,
            "tenacity": mlp_classifier.tenacity,
            "max_epoch": mlp_classifier.max_epoch,
            "actual_epoch": mlp_classifier.nepoch,
            "test_pred": np.array(p_test).tolist(),
            "test_true": np.array(y_true).tolist(),
            "val_pred": np.array(p_val).tolist(),
            "val_true": np.array(y_val_true).tolist(),
            "model_param_names": str([x for x in mlp_classifier.model.named_modules()][0][1])
        }

    def run(self):
        """Main execution flow"""
        x_files = [self.args.xfile] if self.args.xfile else os.listdir(self.args.xpath)
        y_files = os.listdir(self.args.ypath)

        for x in x_files:
            for y in y_files:
                # Skip invalid files
                if not self._validate_files(x, y):
                    continue

                # Clean up GPU memory
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()

                x_name = x[:-4]
                y_name = y[:-5] if y.endswith('.json') else y[:-4]
                print(f"Running on: {x}, {y}")

                # Load and process data
                X, Y = self.load_and_process_data(
                    os.path.join(self.args.xpath, x),
                    os.path.join(self.args.ypath, y)
                )

                X = self.reshape_x(X, x_name)

                # Handle different task types
                if self.args.task_type == 'tokenizer_specific_labels':
                    X, Y = self.process_annotations(X, Y, x_name)
                else:
                    Y = Y.reshape(self.args.data_len, -1)

                # Get dimensions
                if self.args.task_type == 'universal_labelsndard':
                    _, _, embedding_dimension = X.shape
                    _, output_dimension = Y.shape
                else:
                    _, _, embedding_dimension = list(X.values())[0].shape
                    _, output_dimension = list(Y.values())[0].shape

                assert output_dimension > 1

                # Split data
                X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, Y)

                # Train and evaluate
                result = self.train_and_evaluate(
                    X_train, X_val, X_test, y_train, y_val, y_test,
                    embedding_dimension, output_dimension
                )

                # Add metadata to results
                result.update({
                    "X": x_name,
                    "Y": y_name,
                })

                if self.args.num_heads > 1:
                    result["prob_model_variant"] = f"multi_head{self.args.num_heads}"

                # Save results
                self._save_results(result, x_name, y_name)

    def _validate_files(self, x: str, y: str) -> bool:
        """Validate file extensions"""
        if self.args.task_type not in ['tokenizer_specific_labels']:
            return (x.endswith('.npy') and y.endswith('.npy') and
                    ('bucket' in y or self.args.task_type == 'universal_labels_multihead'))
        else:  # coref
            return ((x.endswith('.npy') or x.endswith('.npz')) and
                    y.endswith('.json'))

    def _save_results(self, result: Dict, x_name: str, y_name: str):
        """Save results to JSON file"""
        epoch_str = str(result["actual_epoch"])
        truncate_statement = "_truncateYES" if self.args.truncate else ""

        if self.args.num_heads > 1:
            filename = (f'probresult_multi_head{self.args.num_heads}_{y_name}_{x_name}_'
                        f'epoch{epoch_str}_nhid{str(self.args.nhid)}.json')
        else:
            filename = (f'probresult_{y_name}_{x_name}_epoch{epoch_str}_'
                        f'nhid{str(self.args.nhid)}{truncate_statement}.json')

        file_path = os.path.join(self.args.output_folder, filename)
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description='Probing script')

    # Required arguments
    parser.add_argument('--xpath', required=True, help='Path to X data directory')
    parser.add_argument('--ypath', required=True, help='Path to Y data directory')
    parser.add_argument('--nhid', type=int, required=True, help='Hidden layer size')

    # Optional arguments
    parser.add_argument('--task_type', choices=['universal_labels', 'tokenizer_specific_labels', 'universal_labels_multihead'],
                        default='standard', help='Type of probing task')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='Number of attention heads for multihead variant')
    parser.add_argument('--xfile', help='Specific X file to process')
    parser.add_argument('--output_folder', default='.',
                        help='Output folder for results')
    parser.add_argument('--truncate', action='store_true',
                        help='Whether to truncate sequences')
    parser.add_argument('--truncate_len', type=int, default=512,
                        help='Length to truncate sequences to')
    parser.add_argument('--train_len', type=int, default=1300,
                        help='Number of training examples. 1300 for MUC, 206 for WikiEvents. Expect all embeddings and labels in: order of training, dev, test')
    parser.add_argument('--dev_len', type=int, default=200,
                        help='Number of validation examples. 200 for MUC, 20 for WikiEvents')
    parser.add_argument('--test_len', type=int, default=200,
                        help='Number of test examples. 200 for MUC, 20 for WikiEvents')
    parser.add_argument('--max_epoch', type=int, default=200,
                        help='Maximum number of epochs')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Run probing task
    probing = ProbingTask(args)
    probing.run()