import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import os

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(device)

image_path = []
labels = []
root = Path("./animal-faces/afhq")
for i in root.rglob("*.jpg"):
    image_path.append(str(i))
    labels.append(i.parent.name)

data_df = pd.DataFrame(zip(image_path, labels), columns = ["image_path", "labels"])

train = data_df.sample(frac = 0.7)
left = data_df.drop(train.index)
val = left.sample(frac = 0.5)
test = left.drop(val.index)

label_encoder = LabelEncoder()
label_encoder.fit(data_df['labels'])
print(label_encoder.classes_)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

class ImageDataset(Dataset):
  def __init__(self, dataframe, transform = None):
    self.dataframe = dataframe
    self.transform = transform
    self.label = torch.tensor(label_encoder.transform(dataframe['labels'])).to(device)

  def __len__(self):
    return self.dataframe.shape[0]

  def __getitem__(self, idx):
    img_path = self.dataframe.iloc[idx, 0]
    label = self.label[idx]

    image = Image.open(img_path).convert("RGB")
  
    if self.transform:
      image = self.transform(image).to(device)
  
    return image, label
  
train_dataset = ImageDataset(dataframe = train, transform = transform)
validation_dataset = ImageDataset(dataframe = val, transform = transform)
test_dataset = ImageDataset(dataframe = test, transform = transform)  

lr = 1e-4
BATCH_SIZE = 16
EPOCH = 10
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
validation_dataloader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
    self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)

    self.pooling = nn.MaxPool2d(2, 2)

    self.relu = nn.ReLU()

    self.flatten = nn.Flatten()
    self.linear = nn.Linear((128*16*16), 128)

    self.output = nn.Linear(128, len(data_df['labels'].unique()))
  
  def forward(self,x):
    x = self.conv1(x)
    x = self.pooling(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.pooling(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.pooling(x)
    x = self.relu(x)

    x = self.flatten(x)
    x = self.linear(x)
    x = self.output(x)

    return x

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = lr)

for epoch in range(EPOCH):
    train_acc = 0
    validation_acc = 0
    total_loss_train = 0
    total_loss_val = 0
    testing_acc = 0
    for data in train_dataloader:
        input, label = data
        prediction = model(input).squeeze(1)
        train_loss = criterion(prediction, label)
        total_loss_train += train_loss.item()
        acc = (torch.argmax(prediction, dim=1) == label).sum().item()
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
            acc = (torch.argmax(prediction, dim=1) == label).sum().item()
            validation_acc+= acc
    with torch.no_grad():
        for data in test_dataloader:
            input, label = data
            prediction = model(input).squeeze(1)
            acc = (torch.argmax(prediction, dim=1) == label).sum().item()
            testing_acc+= acc
    print(f"Epoch: {epoch}        train_loss: {round(total_loss_train/train_dataloader.__len__(),4)}        val_loss: {round(total_loss_val/validation_dataloader.__len__(), 4)}        train_acc_rate: {round(train_acc/train_dataset.__len__(), 4)}       val_acc_rate: {round(validation_acc/validation_dataset.__len__(), 4)}       test_acc: {testing_acc/test_dataset.__len__():.4f}")

torch.save(model.state_dict(), "./animal-faces/animal_faces_model.pth")