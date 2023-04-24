import gc
import json
import os

from senteval_classifier import *

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 2000
    probing_classifier_width = 200
    for x in os.listdir("../X/"):
        for y in os.listdir("../Y/"):
            if x[-4:] != ".npy" or y[-4:] != ".npy":
                continue

            print("Running on:", x, y)
            x_name = x[:-4]
            y_name = y[:-4]
            print("loading X Y")
            X = np.load(
                f"../X/{x_name}.npy", allow_pickle=True)
            Y = np.load(
                f"../Y/{y_name}.npy")

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
                    new_X.append(example_X.reshape(-1))
                X = new_X
                max_length = max([len(x) for x in X])
                for i in range(len(X)):
                    if len(X[i]) < max_length:
                        X[i] = np.pad(X[i], (0, max_length - len(X[i])), 'constant', constant_values=(0))
                X = np.array(X)
            Y = Y.reshape(1700, -1)
            X = X.reshape(1700, -1)

            gc.collect()
            # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            X_train, X_val, X_test, y_train, y_val, y_test = X[400:], X[200:400], X[:200], Y[400:], Y[200:400], Y[:200]

            # print("Scale-fitting X Y")
            # scaler = StandardScaler()
            # X_train = scaler.fit_transform(X_train)
            # X_val = scaler.fit_transform(X_val)
            # X_test = scaler.transform(X_test)

            _, input_dimension = X_train.shape
            _, output_dimension = Y.shape

            params = {"nhid": probing_classifier_width, "optim": "adam", "tenacity": 5, "batch_size": 8, "dropout": 0.1,
                      "max_epoch": 2000}
            mlp_classifier = MLP(params, input_dimension, output_dimension, cudaEfficient=device == "cpu")

            mlp_classifier.fit(X_train, y_train, (X_val, y_val), early_stop=True)

            """evaluate model"""

            with torch.no_grad():
                p_train = mlp_classifier.predict(X_train)
                train_labels = y_train
                train_acc = np.mean(train_labels == p_train)

                p_val = mlp_classifier.predict(X_val)
                val_labels = y_val
                val_acc = np.mean(val_labels == p_val)

                p_test = mlp_classifier.predict(X_test)
                test_labels = y_test
                test_acc = np.mean(test_labels == p_test)

            print("train_acc", train_acc)
            print("test_acc", test_acc)
            print("val_acc", val_acc)
            epoch_str = str(epoch)

            # torch.save(model.state_dict(), f"./{y_name}_{x_name}_epoch{epoch_str}.pt")

            y_pred = []
            y_true = []

            y_val_pred = []
            y_val_true = []

            # iterate over test data
            for output, labels in [(p_test[i], y_test[i]) for i in range(len(X_test))]:
                # Feed Network
                y_pred.append(output)  # Save Prediction
                y_true.append(labels)  # Save Truth
                pass

            for output, labels in [(p_val[i], y_val[i]) for i in range(len(X_val))]:
                y_val_pred.append(output)  # Save Prediction
                y_val_true.append(labels)  # Save Truth
                pass

            result = {"test_pred": np.array(y_pred).tolist(),
                      "test_true": np.array(y_true).tolist(),
                      "train_acc": float(train_acc),
                      "val_pred": np.array(y_val_pred).tolist(),
                      "val_true": np.array(y_val_true).tolist(),
                      "val_acc": float(val_acc),
                      "test_acc": float(test_acc), "X": x_name, "Y": y_name}
            print(result)

            with open(f'result_{y_name}_{x_name}_epoch{epoch_str}.json', 'w') as f:
                json.dump(result, f, indent=4)
            pass
