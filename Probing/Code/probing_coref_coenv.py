import gc
import json
import os

from tqdm import tqdm

from senteval_classifier import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    torch.manual_seed(11)
    np.random.seed(11)
    torch.cuda.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    probing_classifier_width = int(os.getenv("nhid"))
    params = {"max_epoch": 200, "nhid": probing_classifier_width, "optim": "adam", "tenacity": 10, "batch_size": 8,
              "dropout": 0.0}
    xpath = os.getenv("x")
    annotation_path = "../XY/"  # os.getenv("annotation")
    for x in os.listdir(xpath):
        for y in os.listdir(annotation_path):
            if x[-4:] != ".npy" or y[-5:] != ".json":
                continue

            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

            x_name = x[:-4]
            y_name = y[:-5]
            print("Running on:", x, y)

            print("loading X Y")
            X = np.load(
                f"{xpath}{x_name}.npy", allow_pickle=True)
            annotation = json.load(open(f"{annotation_path}{y_name}.json"))

            print("Reshaping X Y")
            # Assume each example does not have a dimension of 1, and have more than example
            if "dygie" in x_name or (not hasattr(X, "shape")):
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
                max_length = max([len(x) for x in X])
                for i in range(len(X)):
                    if len(X[i]) < max_length:
                        X[i] = np.pad(X[i], ((0, max_length - len(X[i])), (0, 0)), 'constant', constant_values=(0))
                X = np.array(X)

            for model_name in sorted(list(annotation.keys()), key=len, reverse=True):
                if model_name in x_name.lower():
                    annotation = annotation[model_name]
                    break
            Y = dict()
            X_annotated = dict()

            print("Processing Annotations")
            for partition in annotation.keys():
                for example in tqdm(annotation[partition], desc=partition):
                    l = []
                    idx = 0
                    while f"index{idx}" in example:
                        l.append(X[example['doc_i']][example[f"index{idx}"]])
                        idx += 1
                    if partition not in X_annotated:
                        X_annotated[partition] = np.array([l])
                        Y[partition] = np.array([[0, 1] if example['label'] == 1 else [0, 1]])
                    else:
                        X_annotated[partition] = np.concatenate((X_annotated[partition], np.array([l])))
                        Y[partition] = np.concatenate((Y[partition], [[0, 1] if example['label'] == 1 else [1, 0]]))

            _, _, embedding_dimension = list(X_annotated.values())[0].shape
            _, output_dimension = list(Y.values())[0].shape

            assert output_dimension > 1

            X_train, X_val, X_test, y_train, y_val, y_test = X_annotated['train'], X_annotated['dev'], X_annotated[
                'test'], Y['train'], Y['dev'], Y['test']

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
            with open(f'probresult_{y_name}_{x_name}_epoch{epoch_str}_nhid{str(probing_classifier_width)}.json',
                      'w') as f:
                json.dump(result, f, indent=4)
            pass
