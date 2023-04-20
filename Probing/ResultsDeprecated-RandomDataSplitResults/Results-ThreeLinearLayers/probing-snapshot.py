import json
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def configure_loss_function():
    return torch.nn.MSELoss()
    # return torch.nn.L1Loss()


def configure_optimizer(model):
    return torch.optim.Adam(model.parameters())
    # return torch.optim.SGD(model.parameters(), lr=0.01)


def full_gd(model, criterion, optimizer, X_train, y_train, n_epochs):
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    for it in range(n_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs_test = model(X_test)
        loss_test = criterion(outputs_test, y_test)

        train_losses[it] = loss.item()
        test_losses[it] = loss_test.item()

        if (it + 1) % 50 == 0:
            print(
                f'In this epoch {it + 1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {loss_test.item():.4f}')

    return train_losses, test_losses


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 5000
    for x in os.listdir("../X/"):
        for y in os.listdir("../Y/"):
            gc.collect()
            if x[-4:] != ".npy" or y[-4:] != ".npy":
                continue
            if "_0_" not in x and "_last_"  not in x and "dygie" not in x:
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
            Y = Y.reshape(-1, 1)
            X = X.reshape(1700,-1)
            
            gc.collect()
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

            print("Scale-fitting X Y")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            print("Torch-loading X Y")
            X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
            X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
            y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
            y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

            _, input_dimension = X_train.shape
            _, output_dimension = Y.shape

            print("Creating model")
            #model = torch.nn.Linear(input_dimension, output_dimension)
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dimension,256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 16), 
                torch.nn.ReLU(), 
                torch.nn.Linear(16, output_dimension)
                )
            model = model.to(device)

            print("Training X Y")
            criterion = configure_loss_function()
            optimizer = configure_optimizer(model)
            train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, epoch)

            #plt.plot(train_losses, label='train loss')
            #plt.plot(test_losses, label='test loss')
            #plt.legend()
            #plt.show()

            """evaluate model"""

            with torch.no_grad():
                p_train = np.rint(model(X_train).cpu().detach().numpy()).astype(int)
                p_train = [max(0, p) for p in p_train]
                train_labels = np.rint(y_train.cpu().detach().numpy()).astype(int)
                train_acc = np.mean(train_labels == p_train)

                p_test = model(X_test).cpu().detach().numpy().astype(int)
                p_test = [max(0, p) for p in p_test]
                test_labels = np.rint(y_test.cpu().detach().numpy()).astype(int)
                test_acc = np.mean(test_labels == p_test)

            print("train_acc", train_acc)
            print("test_acc", test_acc)
            epoch_str = str(epoch)

            torch.save(model.state_dict(), f"./{y_name}_{x_name}_epoch{epoch_str}.pt")

            y_pred = []
            y_true = []

            # iterate over test data
            for inputs, labels in [(X_test[i], y_test[i]) for i in range(len(X_test))]:
                output = model(inputs)  # Feed Network
                y_pred.extend(output.cpu().detach().numpy())  # Save Prediction
                labels = labels.data.cpu().detach().numpy()
                y_true.extend(labels)  # Save Truth
                pass

            result = {"y_pred": [float(y) for y in y_pred],
                      "y_pred_int": [max(int(np.rint(y).astype(int)), 0) for y in y_pred],
                      "y_true": [int(y.astype(int)) for y in y_true], "train_acc": float(train_acc),
                      "test_acc": float(test_acc), "X": x_name, "Y": y_name}
            print(result)

            with open(f'result_{y_name}_{x_name}_epoch{epoch_str}.json', 'w') as f:
                json.dump(result, f, indent=4)
            pass
