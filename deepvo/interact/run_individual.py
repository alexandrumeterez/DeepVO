import torch
from deepvo.model.deepvo import DeepVO
from deepvo.data.dataset import KITTIVisualOdometryDataset
from deepvo.conf.params import *
from torchvision import transforms
import os
from torch.autograd import Variable
from deepvo.visualization.routes import plot_both
from deepvo.utils.utils import getPoseFromOdometry
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    model = DeepVO().to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((384, 1280)),
        transforms.CenterCrop((384, 1280)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[127. / 255., 127. / 255., 127. / 255.],
            std=[1 / 255., 1 / 255., 1 / 255.]
        )
    ])

    postprocess = transforms.Normalize(
        mean=[-127., -127., -127.],
        std=[255., 255., 255.]
    )
    dataset = KITTIVisualOdometryDataset(IMAGES_DIR, POSES_DIR, TRAIN_SEQUENCES, TRAJECTORY_LENGTH, preprocess)

    # Load model
    path = os.path.join(SAVED_MODELS_DIR, 'model_6.pth')

    x_est, y_est = [], []
    x_gt, y_gt = [], []

    # Test
    with torch.no_grad():
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        for i in tqdm(range(dataset.size - 1)):
            images, odometries = dataset[i]
            images = torch.FloatTensor(images).to(DEVICE)
            images = images.unsqueeze(0)
            estimations = model(images)
            estimations = estimations.squeeze(0).cpu().detach().numpy()

            # pose = getPoseFromOdometry(estimations)
            # gt = getPoseFromOdometry(odometries)
            pose = estimations[0]
            gt = odometries[0]

            if i > 0:
                x_est.append(pose[0] + x_est[-1])
                y_est.append(pose[2] + y_est[-1])
                x_gt.append(gt[0] + x_gt[-1])
                y_gt.append(gt[2] + y_gt[-1])
            else:
                x_est.append(pose[0])
                y_est.append(pose[2])
                x_gt.append(gt[0])
                y_gt.append(gt[2])
            del images, odometries, pose, estimations

    plt.figure()
    plt.plot(x_est, y_est, color='r', label='Estimation')
    plt.plot(x_gt, y_gt, color='g', label='Ground truth')
    plt.legend()
    plt.show()
