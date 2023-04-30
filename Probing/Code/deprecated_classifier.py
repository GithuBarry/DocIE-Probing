import numpy as np
import torch


def configure_loss_function():
    return torch.nn.MSELoss()
    # return torch.nn.L1Loss()


def configure_optimizer(model):
    return torch.optim.Adam(model.parameters())
    # return torch.optim.SGD(model.parameters(), lr=0.01)


def full_gd(model, criterion, optimizer, X_train, y_train, X_val, y_val, n_epochs):
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
