import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

data_df = pd.read_csv("./riceClassification/riceClassification.csv")
data_df.dropna(inplace = True)
data_df.drop(['id'], axis = 1, inplace = True)

for columns in data_df.columns:
  data_df[columns] = (data_df[columns] - data_df[columns].min())/(data_df[columns].max() - data_df[columns].min())

X = np.array(data_df.iloc[:, :-1])
y = np.array(data_df.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

class dataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype = torch.float32).to(device)
    self.y = torch.tensor(y, dtype = torch.float32).to(device)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    return self.X[index], self.y[index]

training_data = dataset(X_train, y_train)
validation_data = dataset(X_val, y_val)
testing_data = dataset(X_test, y_test)

training_dataloader = DataLoader(training_data, batch_size = 16, shuffle = True)
validation_dataloader = DataLoader(validation_data, batch_size = 16, shuffle = False)
testing_dataloader = DataLoader(testing_data, batch_size = 16, shuffle = False)

HIDDENLAYER = 10
class MyModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_layer = nn.Linear(X.shape[1], HIDDENLAYER)
    self.linear = nn.Linear(HIDDENLAYER, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, X):
    X = self.input_layer(X)
    X = self.linear(X)
    X = self.sigmoid(X)
    return X
model = MyModule().to(device)

criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr = 1e-3)

epochs = 10
for epoch in range(epochs):
    train_acc = 0
    validation_acc = 0
    total_loss_train = 0
    total_loss_val = 0
    testing_acc = 0
    for data in training_dataloader:
        input, label = data
        prediction = model(input).squeeze(1)
        train_loss = criterion(prediction, label)
        total_loss_train += train_loss.item()
        acc = ((prediction).round() == label).sum().item()
        train_acc+= acc
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    with torch.no_grad():
        for data in validation_dataloader:
            input, label = data
            prediction = model(input).squeeze(1)
            val_loss = criterion(prediction, label)
            total_loss_val += val_loss.item()
            acc = ((prediction).round() == label).sum().item()
            validation_acc+= acc
    with torch.no_grad():
        for data in testing_dataloader:
            input, label = data
            prediction = model(input).squeeze(1)
            acc = ((prediction).round() == label).sum().item()
            testing_acc+= acc
    print(f"Epoch: {epoch}        train_loss: {round(total_loss_train/training_dataloader.__len__(),4)}        val_loss: {round(total_loss_val/validation_dataloader.__len__(), 4)}        train_acc_rate: {round(train_acc/training_data.__len__(), 4)}       val_acc_rate: {round(validation_acc/validation_data.__len__(), 4)}       test_acc: {testing_acc/testing_data.__len__():.4f}")