from data_loader import datasets
from models import resnet
from train_utils import trainer
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


train_transform = A.Compose(
                            [
                            A.Resize(300, 300),
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                            A.RandomBrightnessContrast(p=0.5),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2(),
                            ]
                            )

valid_transform = A.Compose(
                            [
                            A.Resize(300, 300),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2(),
                            ]
                            )

train_dataset = datasets.DogsVsCatsDataset("/home/lovekesh/Documents/nnet_experiments/dogsvscats/dataset/train/images.txt", 
                            transforms=train_transform)
train_data = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, shuffle=True)

valid_dataset = datasets.DogsVsCatsDataset("/home/lovekesh/Documents/nnet_experiments/dogsvscats/dataset/train/images.txt", 
                            transforms=valid_transform)
valid_data = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, shuffle=True)

net = resnet.Resnet34()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
trainer.train(net, train_data, 5, optimizer, loss_fn)
