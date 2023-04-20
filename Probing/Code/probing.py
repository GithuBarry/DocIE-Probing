import json
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler



def configure_loss_function():
    return torch.nn.MSELoss()
    # return torch.nn.L1Loss()


def configure_optimizer(model):
    return torch.optim.Adam(model.parameters())
    # return torch.optim.SGD(model.parameters(), lr=0.01)


def full_gd(model, criterion, optimizer, X_train, y_train, X_val, y_val,n_epochs):
    train_losses = np.zeros(n_epochs)
    val_losses = np.zeros(n_epochs)

    for it in range(n_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs_test = model(X_val)
        loss_test = criterion(outputs_test, y_val)

        train_losses[it] = loss.item()
        val_losses[it] = loss_test.item()
        if (it + 1) % 50 == 0:
            print(
                f'In this epoch {it + 1}/{n_epochs}, Training loss: {loss.item():.4f}, Val loss: {loss_test.item():.4f}')
    return train_losses, val_losses


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 2000
    for x in os.listdir("../X/"):
        for y in os.listdir("../Y/"):
            gc.collect()
            if "bucket" not in y:
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
            X = X.reshape(1700,-1)
            
            gc.collect()
            #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            X_train, X_val, X_test, y_train, y_val, y_test = X[400:], X[200:400], X[:200], Y[400:], Y[200:400], Y[:200]

            #print("Scale-fitting X Y")
            #scaler = StandardScaler()
            #X_train = scaler.fit_transform(X_train)
            #X_test = scaler.transform(X_test)
            print("Scale-fitting X Y")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.fit_transform(X_val)
            X_test = scaler.transform(X_test)

            print("Torch-loading X Y")
            X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
            X_val = torch.from_numpy(X_val.astype(np.float32)).to(device)
            X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
            y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
            y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)
            y_val = torch.from_numpy(y_val.astype(np.float32)).to(device)

            _, input_dimension = X_train.shape
            _, output_dimension = Y.shape

            print("Creating model")
            model = torch.nn.Linear(input_dimension, output_dimension)
            #model = torch.nn.Sequential(
            #    torch.nn.Linear(input_dimension,256),
            #    torch.nn.ReLU(),
            #    torch.nn.Linear(256, output_dimension)
            #    )
            model = model.to(device)

            print("Training X Y")
            criterion = configure_loss_function()
            optimizer = configure_optimizer(model)
            train_losses, val_loss = full_gd(model, criterion, optimizer, X_train, y_train, X_val, y_val, epoch)

            #plt.plot(train_losses, label='train loss')
            #plt.plot(test_losses, label='test loss')
            #plt.legend()
            #plt.show()
            plt.plot(train_losses, label='train loss')
            plt.plot(val_loss, label='val loss')
            plt.legend()
            # plt.show()

            """evaluate model"""

            def round_up(p, labels):
                if output_dimension==1:     
                    p = np.rint(p)
                    p = [max(0, pi) for pi in p]
                    labels = labels.astype(int)
                else:
                    p = [np.where(x == max(x))[0] for x in p]
                    labels = [np.where(x == 1)[0] for x in labels.astype(int)]
                    print("p,l",p,labels)
                return np.array(p),np.array(labels)

            with torch.no_grad():
                p_train = model(X_train).cpu().detach().numpy()
                train_labels = y_train.cpu().detach().numpy()
                p_test, train_labels = round_up(p_train, train_labels)
                train_acc = np.mean(train_labels == p_train)

                p_val = model(X_val).cpu().detach().numpy()
                val_labels = y_val.cpu().detach().numpy()
                p_val,val_labels = round_up(p_val, val_labels )
                val_acc = np.mean(val_labels == p_val)

                p_test = model(X_test).cpu().detach().numpy()
                test_labels = y_test.cpu().detach().numpy()
                p_test,test_labels= round_up(p_test,test_labels)
                test_acc = np.mean(test_labels == p_test)

            print("train_acc", train_acc)
            print("test_acc", test_acc)
            print("val_acc", val_acc)
            epoch_str = str(epoch)

            torch.save(model.state_dict(), f"./{y_name}_{x_name}_epoch{epoch_str}.pt")

            y_pred = []
            y_true = []

            y_val_pred = []
            y_val_true = []

            # iterate over test data
            for inputs, labels in [(X_test[i], y_test[i]) for i in range(len(X_test))]:
                output = model(inputs)  # Feed Network
                y_pred.append(output.cpu().detach().numpy())  # Save Prediction
                labels = labels.data.cpu().detach().numpy()
                y_true.append(labels)  # Save Truth
                pass

            for inputs, labels in [(X_val[i], y_val[i]) for i in range(len(X_val))]:
                output = model(inputs)  # Feed Network
                y_val_pred.append(output.cpu().detach().numpy())  # Save Prediction
                labels = labels.data.cpu().detach().numpy()
                y_val_true.append(labels)  # Save Truth
                pass

            result = {"test_pred": np.array(y_pred).tolist(),
                      "test_true":  np.array(y_true).tolist(),
                      "train_acc": float(train_acc),
                      "val_pred": np.array(y_val_pred).tolist(),
                      "val_true": np.array(y_val_true).tolist(),
                      "val_acc": float(val_acc),
                      "test_acc": float(test_acc), "X": x_name, "Y": y_name}
            print(result)

            with open(f'result_{y_name}_{x_name}_epoch{epoch_str}.json', 'w') as f:
                json.dump(result, f, indent=4)
            pass
