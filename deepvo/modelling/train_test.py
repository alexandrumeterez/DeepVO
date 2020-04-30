import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from deepvo.conf.params import *
from deepvo.model.deepvo import DeepVO
from torch.utils.data import DataLoader
from deepvo.data.dataset import KITTIVisualOdometryDataset
from torchvision import transforms
import time

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    n_batches = len(train_loader)
    for batch, (images, odometries) in enumerate(train_loader):
        start = time.time()
        images = Variable(images).float().to(DEVICE)
        odometries = Variable(odometries).float().to(DEVICE)
        estimation = model(images)
        loss = criterion(estimation[:, :, :3], odometries[:, :, :3]) + K * criterion(estimation[:, :, 3:],
                                                                                     odometries[:, :, 3:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        end = time.time()
        print(f'\t Batch [{batch}/{n_batches}], {end-start}s -> Loss: {loss.item()}')
    return epoch_loss / n_batches


if __name__ == '__main__':
    model = DeepVO().to(DEVICE)
    preprocess = transforms.Compose([
        transforms.Resize((384, 1280)),
        transforms.CenterCrop((384, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[127. / 255., 127. / 255., 127. / 255.],
            std=[1 / 255., 1 / 255., 1 / 255.]
        )
    ])
    dataset = KITTIVisualOdometryDataset(IMAGES_DIR, POSES_DIR, TRAIN_SEQUENCES, TRAJECTORY_LENGTH,
                                         transform=preprocess)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch}/{EPOCHS}')
        epoch_loss = train_model(model, train_loader, criterion, optimizer)
        print(f'Final epoch loss: {epoch_loss}')
