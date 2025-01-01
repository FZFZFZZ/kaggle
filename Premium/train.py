import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),  # Input -> Hidden Layer
            nn.ReLU(),                # Activation
            nn.Linear(32, 16),  # Input -> Hidden Layer
            nn.ReLU(),                # Activation
            nn.Linear(16, 8),  # Input -> Hidden Layer
            nn.ReLU(),                # Activation
            nn.Linear(8, 1),  # Input -> Hidden Layer
        )
    
    def forward(self, x):
        return self.model(x)


def train_and_evaluate_model(data):
    device = torch.device('cpu')
    
    y = data['P_Premium Amount'].values
    X = data.drop(columns=['P_Premium Amount']).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    print(X_train.shape)
    input_dim = X_train.shape[1]
    model = RegressionNN(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    max_epochs = 10000
    mse_per_epoch = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_tensor).squeeze()
            mse = mean_squared_error(y_train, y_pred_train.numpy())
            mse_per_epoch.append(mse)
            logger.info(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.4f}, MSE: {mse:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).squeeze()
        mse = mean_squared_error(y_test, y_pred_test.numpy())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test.numpy())
        r2 = r2_score(y_test, y_pred_test.numpy())

        logger.info(f"Final Test MSE: {mse}")
        logger.info(f"Final Test RMSE: {rmse}")
        logger.info(f"Final Test MAE: {mae}")
        logger.info(f"Final Test R^2: {r2}")

    return model




