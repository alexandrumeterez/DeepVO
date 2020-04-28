import os
import numpy as np
from torch.utils.data import Dataset
from deepvo.conf.params import *
from deepvo.utils.utils import load_image, Log, get6DoFPose, visualize

TAG = 'KITTIVisualOdometryDataset'


class KITTIVisualOdometryDataset(Dataset):
    def __init__(self, images_dir, poses_dir, train_sequences, trajectory_length=10, transform=None):
        self.images_dir = images_dir
        self.poses_dir = poses_dir
        self.sequences = train_sequences
        self.size = 0  # Dataset size, with all the poses/images
        self.sequences_sizes = []  # Array with size for each sequence
        self.poses = self.load_poses()
        self.transform = transform
        self.trajectory_length = trajectory_length

    def load_poses(self):
        poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.poses_dir, sequence + '.txt')) as f:
                sequence_poses = np.array([[float(x) for x in line.split()] for line in f])
            poses.append(sequence_poses)
            self.size += len(sequence_poses)  # Sum to total size
            self.sequences_sizes.append(len(sequence_poses))
        return poses

    def load_image(self, sequence, index):
        # Log(TAG, f'Loading {sequence} : {index}')
        path = os.path.join(self.images_dir, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
        image = np.array(load_image(path))
        image = np.moveaxis(image, 2, 0)
        return image

    def get_sequence(self, index):
        total = 0
        for sequence, seq_size in enumerate(self.sequences_sizes):
            # If we can fit [index, index + traj_len] in this sequence,
            # then choose this sequence
            if index + self.trajectory_length < total:
                return sequence, total - index
            total += seq_size

    def __getitem__(self, index):
        # Find sequence that contains index and the index corespondent in the sequence
        sequence, index = self.get_sequence(index)

        # Form a batch
        imgs_batch, odom_batch = [], []
        for i in range(index, index + self.trajectory_length):
            img1 = self.load_image(self.sequences[sequence], i)
            img2 = self.load_image(self.sequences[sequence], i + 1)
            odom1 = get6DoFPose(self.poses[sequence][i])
            odom2 = get6DoFPose(self.poses[sequence][i + 1])
            odom = odom2 - odom1
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            imgs_batch.append(np.concatenate([img1, img2], axis=0))
            odom_batch.append(odom)
        return np.array(imgs_batch), np.array(odom_batch)


# Test
if __name__ == '__main__':
    vods = KITTIVisualOdometryDataset(IMAGES_DIR, POSES_DIR, TRAIN_SEQUENCES)
    imgs, odom = vods[5000]
    visualize(imgs, vods.trajectory_length)

    print(imgs.shape)
    print(odom.shape)
