import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from skmultilearn.problem_transform import ClassifierChain
import json
import os
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    X = np.load(
        "/home/zw545/GTT_Probe/model_gtt/X_last_hidden_layer_GTT_20Epoch_bert_muc1700.npy")
    Y = np.load(
        "/home/zw545/GTT_Probe/model_gtt/Y_muc_1700_Input_Len.npy")

    # One Hot Encoding
    enc = OneHotEncoder()

    print(1,X.shape)
    l = []
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    print("X.shape",X.shape)

    
    Y = enc.fit_transform(Y.reshape(-1, 1)).toarray().astype(int)
    #Y = Y.reshape(-1, 1)
    print("Y.shape",X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(X_test.shape)


    class LinearLayerClassification(torch.nn.Module):
        def __init__(self, input_dimension):
            super().__init__()
            self.linear = torch.nn.Linear(input_dimension, 2)

        def forward(self, input_dimension):
            return self.linear(input_dimension)


    _, input_dimension = X_train.shape
    _, output_dimension = Y.shape

    model = torch.nn.Linear(input_dimension, output_dimension)

    """train the model"""


    def configure_loss_function():
        return torch.nn.MSELoss()


    def configure_optimizer(model):
        return torch.optim.Adam(model.parameters())


    def full_gd(model, criterion, optimizer, X_train, y_train, n_epochs=2000):
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


    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    criterion = configure_loss_function()
    optimizer = configure_optimizer(model)
    # train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train)

    # plt.plot(train_losses, label='train loss')
    # plt.plot(test_losses, label='test loss')
    # plt.legend()
    # plt.show()
    #model = torch.load(open(
        #"/Users/barry/Library/Mobile Documents/com~apple~CloudDocs/Programming/probing/TANL-MucV0-Input-Len-One-Hot.pt"))

    """evaluate model"""

    with torch.no_grad():
        p_train = model(X_train)
        p_train = (p_train.numpy() > 0)

        train_acc = np.mean(y_train.numpy() == p_train)

        p_test = model(X_test)
        p_test = (p_test.numpy() > 0)

        test_acc = np.mean(y_test.numpy() == p_test)

    print(train_acc)
    print(test_acc)

    torch.save(model.state_dict(), "./new_name.pt")

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in [(X_test[i], y_test[i]) for i in range(len(X_test))]:
        output = model(inputs)  # Feed Network

        # output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    # classes = sorted(list(set(y_true).union(set(y_pred))))

    # Build confusion matrix
    # cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix = confusion_matrix([enc.inverse_transform(y) for y in y_true],
                                 np.array([enc.inverse_transform(y.detach()) for y in y_pred]).astype(np.int),
                                 labels=enc.categories_[0])
    df_cm = pd.DataFrame(cf_matrix, index=enc.categories_[0],
                         columns=enc.categories_[0])
    plt.figure(figsize=(12, 11))
    plt.xlabel = "Pred"
    plt.ylabel = "True"
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output_entities.png')
