import torch
import torch.nn as nn
import torch.optim as optim

from model import LeakNet


def weighted_mse(pred, target):
    weights = 1.0 + 5.0 * torch.abs(target)
    return torch.mean(weights * (pred - target) ** 2)


def train_model(X_train, y_train, X_val, y_val, epochs=200, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeakNet(X_train.shape[1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        pred_train = model(X_train_t)
        loss_train = weighted_mse(pred_train, y_train_t)

        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            loss_val = weighted_mse(pred_val, y_val_t)

        train_losses.append(loss_train.item())
        val_losses.append(loss_val.item())

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={loss_train.item():.6f} | "
                f"val_loss={loss_val.item():.6f}"
            )

    return model, train_losses, val_losses