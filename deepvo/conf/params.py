import torch

POSES_DIR = '/home/alex/dataset/poses'
IMAGES_DIR = '/home/alex/dataset/dataset/dataset'
SAVED_MODELS_DIR = '/home/alex/DeepVO/deepvo/models'
# Model params
TRAIN_SEQUENCES = ['03']
TEST_SEQUENCES = ['03']
LEARNING_RATE = 0.001
BATCH_SIZE = 1
TRAJECTORY_LENGTH = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
K = 100  # Per the paper
EPOCHS = 100
