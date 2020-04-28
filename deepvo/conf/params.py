import torch

POSES_DIR = '/home/alex/dataset/poses'
IMAGES_DIR = '/home/alex/dataset/dataset/dataset'

# Model params
TRAIN_SEQUENCES = ['03']
LEARNING_RATE = 0.001
BATCH_SIZE = 16
TRAJECTORY_LENGTH = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
K = 100  # Per the paper
EPOCHS = 100