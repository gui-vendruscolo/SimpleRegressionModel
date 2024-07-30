import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

boston = fetch_california_housing()
X = boston.data
y = boston.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class SimpleRegressor(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
input_dim = X_train.shape[1]
hidden_dim = 50
output_dim = 1

model = SimpleRegressor(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


print("now training")
n_epochs = 200000
for epoch in range(n_epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100000 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

    mean_absolute_error = torch.mean(torch.abs(y_pred - y_test)).item()
    print(f'Mean absolute error: {mean_absolute_error:.4f}')
    
    ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
    ss_residual = torch.sum((y_test - y_pred) ** 2)
    r_squared = 1 - ss_residual / ss_total
    print(f'R-squared: {r_squared:.4f}')

y_pred = y_pred.numpy()
y_test = y_test.numpy()

plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Identity line (Perfect prediction)")
plt.text(0.0, 6, (f'n_epochs = {n_epochs}\nTest Loss = {test_loss:.4f}\nMean absolute error = {mean_absolute_error:.4f}\nR Squared = {r_squared:.4f}'), fontsize = 12)

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()

plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Identity line (Perfect prediction)")

plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()

