import torch.optim as optim
from torch.autograd import Variable
from deepvo.conf.params import *
from deepvo.model.deepvo import DeepVO
from torch.utils.data import DataLoader
from deepvo.data.dataset import KITTIVisualOdometryDataset
from torchvision import transforms
from deepvo.utils.utils import save_model
from tqdm import tqdm


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    pos_loss = 0
    angle_loss = 0
    n_batches = len(train_loader)
    for batch, (images, odometries) in tqdm(enumerate(train_loader), total=n_batches):
        images = Variable(images).float().to(DEVICE)
        odometries = Variable(odometries).float().to(DEVICE)
        estimation = model(images)
        pos_mse = criterion(estimation[:, :, :3], odometries[:, :, :3])
        angle_mse = K * criterion(estimation[:, :, 3:], odometries[:, :, 3:])
        loss = pos_mse + angle_mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pos_loss += pos_mse.item()
        angle_loss += angle_mse.item()
    return epoch_loss / n_batches, pos_loss / n_batches, angle_loss / n_batches


def test_model(model, test_loader, criterion):
    model.eval()
    epoch_loss = 0
    pos_loss = 0
    angle_loss = 0
    with torch.no_grad():
        n_batches = len(test_loader)
        for batch, (images, odometries) in tqdm(enumerate(test_loader), total=n_batches):
            images = Variable(images).float().to(DEVICE)
            odometries = Variable(odometries).float().to(DEVICE)
            estimation = model(images)
            pos_mse = criterion(estimation[:, :, :3], odometries[:, :, :3])
            angle_mse = K * criterion(estimation[:, :, 3:], odometries[:, :, 3:])
            loss = pos_mse + angle_mse
            epoch_loss += loss.item()
            pos_loss += pos_mse.item()
            angle_loss += angle_mse.item()
    return epoch_loss / n_batches, pos_loss / n_batches, angle_loss / n_batches


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
    train_dataset = KITTIVisualOdometryDataset(IMAGES_DIR, POSES_DIR, TRAIN_SEQUENCES, TRAJECTORY_LENGTH,
                                               transform=preprocess)
    test_dataset = KITTIVisualOdometryDataset(IMAGES_DIR, POSES_DIR, TEST_SEQUENCES, TRAJECTORY_LENGTH,
                                              transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch + 1}/{EPOCHS}')

        print(f'Train')
        train_loss, train_pos_loss, train_angle_loss = train_model(model, train_loader, criterion, optimizer)
        print(
            f'\tEpoch loss: {train_loss:.4f}, Position loss: {train_pos_loss:.4f}. Angle loss: {train_angle_loss:.4f}')

        print(f'Test')
        test_loss, test_pos_loss, test_angle_loss = test_model(model, train_loader, criterion)
        print(f'\tEpoch loss: {test_loss:.4f}, Position loss: {test_pos_loss:.4f}. Angle loss: {test_angle_loss:.4f}')

        save_model(model, f'model_{epoch + 1}.pth')
