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

    # Test
    with torch.no_grad():
        model.load_state_dict(torch.load(path, map_location=DEVICE))

        images, odometries = dataset[100]
        del dataset
        images = torch.FloatTensor(images).to(DEVICE)
        images = images.unsqueeze(0)
        estimations = model(images)
        estimations = estimations.squeeze(0).cpu().detach().numpy()
        images = images.squeeze(0).cpu().detach()
        # Deconcat images
        images = images[:, :3, :, :]
        for i in range(TRAJECTORY_LENGTH):
            images[i, :, :, :] = postprocess(images[i, :3, :, :])
        images = images.permute(0, 2, 3, 1)
        images = np.array(images * 255, dtype=np.uint8)
        # Plot
        gt_poses = getPoseFromOdometry(odometries)
        est_poses = getPoseFromOdometry(estimations)
        plot_both(images, TRAJECTORY_LENGTH, gt_poses, est_poses)
