from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import StandardScaler

import json
class LinearLayerClassification(torch.nn.Module, ABC):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(input_dimension, output_dimension)
        # self.rectifier = torch.nn.ReLU()

    def forward(self, input):
        # return self.rectifier((self.linear(input)))
        return self.linear(input)


def configure_loss_function():
    return torch.nn.MSELoss()
    #return torch.nn.L1Loss()


def configure_optimizer(model):
    return torch.optim.Adam(model.parameters())
    #return torch.optim.SGD(model.parameters(), lr=0.01)


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
    epoch = 2000
    for x in ["X_dygiepp_embedding-unpadded_bert-uncased_epoch21_muc1700.npy"]:# os.listdir("../X/"):
        for y in os.listdir("../Y/"):
            if x[-4:] != ".npy" or y[-4:] != ".npy" or "5_" in x or "2_" in x or "3_" in x or "4_" in x or "6_" in x or  "7_" in x or  "8_" in x or  "9_" in x or  "10_" in x or  "11_" in x:
                continue
            print("Running on:",x,y)
            x_name = x[:-4]
            y_name = y[:-4]
            #x_name = "X_TANL_layer_last_bert-uncased_epoch20_muc1700"
            #y_name = "Y_muc_1700_num_events"
            print("loading X Y")
            X = np.load(
                f"../X/{x_name}.npy",allow_pickle=True)
            Y = np.load(
                f"../Y/{y_name}.npy")

            print("Reshaping X Y")
            Y = Y.reshape(-1, 1)
            X = X.reshape(1700, -1)
            lengths = [len(x) for x in X]
            for i in range(len(X)):
                if len(X[i]) < max(lengths):
                    np.append(X[i], [0]* (max(lengths) - len(X[i])))

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

            model = torch.nn.Linear(input_dimension, output_dimension).to(device)

            print("Training X Y")
            criterion = configure_loss_function()
            optimizer = configure_optimizer(model)
            train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train, epoch)

            plt.plot(train_losses, label='train loss')
            plt.plot(test_losses, label='test loss')
            plt.legend()
            plt.show()

            """evaluate model"""

            with torch.no_grad():
                p_train = np.rint(model(X_train).cpu().detach().numpy()).astype(int)
                train_labels = np.rint(y_train.cpu().detach().numpy()).astype(int)
                train_acc = np.mean(train_labels == p_train)

                p_test = model(X_test).cpu().detach().numpy().astype(int)
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

            result = {"y_pred":[float(y) for y in y_pred],"y_pred_int": [int(np.rint(y).astype(int)) for y in y_pred], "y_true": [int(y.astype(int)) for y in y_true], "train_acc":float(train_acc), "test_acc":float(test_acc), "X": x_name, "Y":y_name}
            print(result)
            
            with open(f'result_{y_name}_{x_name}_epoch{epoch_str}.json', 'w') as f:
                json.dump(result, f, indent=4)
            pass

            # constant for classes
            # classes = sorted(list(set(y_true).union(set(y_pred))))

            # Build confusion matrix
            # cf_matrix = confusion_matrix(y_true, y_pred)
            # cf_matrix = confusion_matrix([enc.inverse_transform(y) for y in y_true],
            #                              np.array([enc.inverse_transform(y.detach()) for y in y_pred]).astype(np.int),
            #                              labels=enc.categories_[0])
            # df_cm = pd.DataFrame(cf_matrix, index=enc.categories_[0],
            #                      columns=enc.categories_[0])
            # plt.figure(figsize=(12, 11))
            # plt.xlabel = "Pred"
            # plt.ylabel = "True"
            # sn.heatmap(df_cm, annot=True)
            # plt.savefig('output_entities.png')
