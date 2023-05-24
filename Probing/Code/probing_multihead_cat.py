import gc
import json
import os

from senteval_classifier import *

num_attention_embedding = int(os.getenv("num_attention_embedding"))


class WeightedEmbeddingSumLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(WeightedEmbeddingSumLayer, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.linears = []
        for i in range(num_attention_embedding):
            self.linears.append(nn.Linear(embedding_dim, 1))

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

        # Concatenate all weighted sums along the last dimension
        concatenated_weighted_sums = torch.cat(weighted_sums, dim=-1)
        # concatenated_weighted_sums shape: (batch_size, hidden_size * num_layers)

        return concatenated_weighted_sums


class MLP(PyTorchClassifier):
    def __init__(self, params, inputdim, nclasses, l2reg=0., batch_size=64,
                 seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        if params["nhid"] == 0:
            print("Regression!")
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            ).to(device)
        else:
            self.model = nn.Sequential(
                WeightedEmbeddingSumLayer(inputdim),
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
            ).to(device)

        self.loss_fn = nn.MSELoss().to(device)
        self.loss_fn.size_average = False

        optim_fn, optim_params = get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg


if __name__ == "__main__":

    probing_classifier_width = int(os.getenv("nhid"))
    params = {"max_epoch": 200, "nhid": probing_classifier_width, "optim": "adam", "tenacity": 10, "batch_size": 8,
              "dropout": 0.0}
    xpath = os.getenv("x")
    ypath = os.getenv("y")
    for x in os.listdir(xpath):
        for y in os.listdir(ypath):
            if x[-4:] != ".npy" or y[-4:] != ".npy":
                continue
            if "bucket" not in y:
                continue

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

            print("Running on:", x, y)
            x_name = x[:-4]
            y_name = y[:-4]
            print("loading X Y")
            X = np.load(
                f"{xpath}{x_name}.npy", allow_pickle=True)
            Y = np.load(
                f"{ypath}{y_name}.npy")

            print("Reshaping X Y")
            # Assume each example does not have a dimension of 1, and have more than example
            if "dygie" in x_name:
                pickled_X = X
                new_X = []
                while len(pickled_X) == 1:
                    pickled_X = pickled_X[0]
                for example_X in pickled_X:
                    while len(example_X) == 1:
                        example_X = example_X[0]
                    new_X.append(example_X)
                X = new_X
                max_length = max([len(x) for x in X])
                for i in range(len(X)):
                    if len(X[i]) < max_length:
                        X[i] = np.pad(X[i], ((0, max_length - len(X[i])), (0, 0)), 'constant', constant_values=(0))
                X = np.array(X)
            Y = Y.reshape(1700, -1)
            # X = X.reshape(1700, -1)

            _, _, embedding_dimension = X.shape
            _, output_dimension = Y.shape

            assert output_dimension > 1

            X_train, X_val, X_test, y_train, y_val, y_test = X[400:], X[200:400], X[:200], Y[400:], Y[200:400], Y[:200]

            mlp_classifier = MLP(params=params, inputdim=embedding_dimension, nclasses=output_dimension,
                                 cudaEfficient=not torch.cuda.is_available())

            print("Fitting MLP Classifier")
            mlp_classifier.fit(X_train, y_train, (X_val, y_val), early_stop=True)

            """evaluate model"""

            x_torch, y_torch = torch.from_numpy(X_train).to(device, dtype=torch.float32), torch.from_numpy(y_train).to(
                device, dtype=torch.int64)
            train_acc, p_train = mlp_classifier.score_and_prob(x_torch, y_torch)

            x_torch, y_torch = torch.from_numpy(X_val).to(device, dtype=torch.float32), torch.from_numpy(y_val).to(
                device, dtype=torch.int64)
            val_acc, p_val = mlp_classifier.score_and_prob(x_torch, y_torch)

            x_torch, y_torch = torch.from_numpy(X_test).to(device, dtype=torch.float32), torch.from_numpy(y_test).to(
                device, dtype=torch.int64)
            test_acc, p_test = mlp_classifier.score_and_prob(x_torch, y_torch)

            print("train_acc", train_acc)
            print("test_acc", test_acc)
            print("val_acc", val_acc)
            epoch_str = str(mlp_classifier.nepoch)

            # y_pred = []
            y_true = []

            # y_val_pred = []
            y_val_true = []

            # iterate over test data
            for output, labels in [(p_test[i], y_test[i]) for i in range(len(X_test))]:
                # Feed Network
                # y_pred.append(output)  # Save Prediction
                y_true.append(labels)  # Save Truth
                pass

            for output, labels in [(p_val[i], y_val[i]) for i in range(len(X_val))]:
                # y_val_pred.append(output)  # Save Prediction
                y_val_true.append(labels)  # Save Truth
                pass

            result = {"X": x_name, "Y": y_name,
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
                      "model_param_names": str([x for x in mlp_classifier.model.named_modules()][0][1])}

            print(result)
            # torch.save(mlp_classifier.model.state_dict(), f"./{y_name}_{x_name}_epoch{epoch_str}_nhid{str(probing_classifier_width)}.pt")
            with open(
                    f'probresult_{str(num_attention_embedding)}attentokens_{y_name}_{x_name}_epoch{epoch_str}_nhid{str(probing_classifier_width)}.json',
                    'w') as f:
                json.dump(result, f, indent=4)
            pass
