import torch
import torchvision.transforms as transforms
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
MODEL_TO_TEST = 'model_20251112_155739_6'

DEVICE = torch.device("cpu")
NUM_WORKERS = 8
LOGGING_INTERVAL = 250
L4_LR = 1e-7
FC_LR = 1e-6

MISCLASSIFIED_COUNT = 64
CLASSES = ['Cat', 'Dog']

DATA_FOLDERS = [
    'data/train/cat',
    'data/train/dog',
    'data/valid/cat',
    'data/valid/dog',
]
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png']
SYSTEM_FILES = ['Thumbs.db', '.DS_Store']
BLANK_THRESHOLD = 5.0
IMAGES_PER_ROW = 8