from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("loading X Y")
    X = np.load(
        "../X/X_TANL_layer_last_bert-uncased_epoch20_muc1700.npy")
    Y = np.load(
        "../Y/Y_muc_1700_num_events.npy")

    print("Reshaping X Y")
    Y = Y.reshape(-1, 1)
    X = X.reshape(1700, -1)
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

    model = LinearLayerClassification(input_dimension, output_dimension).to(device)

    print("Training X Y")
    criterion = configure_loss_function()
    optimizer = configure_optimizer(model)
    train_losses, test_losses = full_gd(model, criterion, optimizer, X_train, y_train)

    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()

    """evaluate model"""

    with torch.no_grad():
        p_train = model(X_train).cpu().detach().numpy().astype(int)
        print("train bool:", y_train.cpu().detach().numpy().astype(int) == p_train)
        train_acc = np.mean(y_train.cpu().detach().numpy().astype(int) == p_train)

        p_test = model(X_test).cpu().detach().numpy().astype(int)
        print("test bool:", y_test.cpu().detach().numpy().astype(int) == p_test)
        test_acc = np.mean(y_test.cpu().detach().numpy().astype(int) == p_test)

    print("train_acc", train_acc)
    print("test_acc", test_acc)

    torch.save(model.state_dict(), "./new_name.pt")

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in [(X_test[i], y_test[i]) for i in range(len(X_test))]:
        output = torch.relu(model(inputs))  # Feed Network
        y_pred.extend(output.cpu().detach().numpy())  # Save Prediction
        labels = labels.data.cpu().detach().numpy()
        y_true.extend(labels)  # Save Truth
        pass
    print("y_pred", y_pred)
    print("y_pred_int", [y.astype(int) for y in y_pred])
    print("y_true", y_true)
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
