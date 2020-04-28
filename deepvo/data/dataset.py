import os
import numpy as np
from torch.utils.data import Dataset
from deepvo.conf.params import *
from deepvo.utils.utils import load_image, Log

TAG = 'KITTIVisualOdometryDataset'


class KITTIVisualOdometryDataset(Dataset):
    def __init__(self, images_dir, poses_dir, train_sequences, trajectory_length=10, transform=None):
        self.images_dir = images_dir
        self.poses_dir = poses_dir
        self.sequences = train_sequences
        self.size = 0  # Dataset size, with all the poses/images
        self.sequences_sizes = []  # Array with size for each sequence
        self.poses = self.__load_poses()
        self.transform = transform
        self.trajectory_length = trajectory_length

    def __load_poses(self):
        poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.poses_dir, sequence + '.txt')) as f:
                sequence_poses = np.array([[float(x) for x in line.split()] for line in f])
            poses.append(sequence_poses)
            self.size += len(sequence_poses)  # Sum to total size
            self.sequences_sizes.append(len(sequence_poses))
        return poses

    def __load_image(self, sequence, index):
        # Log(TAG, f'Loading {sequence} : {index}')
        path = os.path.join(self.images_dir, 'sequences', sequence, 'image_2', '%06d' % index + '.png')
        image = load_image(path)
        return image


# Test
if __name__ == '__main__':
    vods = KITTIVisualOdometryDataset(IMAGES_DIR, POSES_DIR, TRAIN_SEQUENCES)
    print(vods.size)
    print(vods.sequences_sizes)
    vods.load_image('00', 10)
