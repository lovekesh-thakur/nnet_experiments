from data_loader import datasets
from models import resnet
from train_utils import trainer
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from config import *
import json

import wandb

wandb.login()

device = "cuda" if torch.cuda.is_available() else "cpu"

conf = adam_config
resume_training = False
resume_checkpoint = None


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

### Dataset Loader
train_dataset = datasets.DogsVsCatsDataset("/home/lovekesh/Developer/nnet_experiments/dogsvscats/data/train.txt", 
                            transforms=train_transform)
train_data = DataLoader(dataset=train_dataset, batch_size=conf['BATCH_SIZE'], num_workers=4, shuffle=True)

valid_dataset = datasets.DogsVsCatsDataset("/home/lovekesh/Developer/nnet_experiments/dogsvscats/data/valid.txt", 
                            transforms=valid_transform)
valid_data = DataLoader(dataset=train_dataset, batch_size=conf['BATCH_SIZE'], num_workers=4, shuffle=True)

### preparing model and loss function
net = resnet.Resnet34().to(device=device)
loss_fn = nn.CrossEntropyLoss()

## setting up optimizer
if conf['optimizer'] == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=conf['learning_rate'], momentum=0.9)
elif conf['optimizer'] == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr = conf['learning_rate'])

if resume_training:
    checkpoint = torch.load(resume_checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
else:
    start_epoch = 0


## setting up learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

best_val_loss = 10000

## running training and logging with wandb
with wandb.init(project = 'resnet34-classifier', config = conf):
    start_iterations = 0
    for i in tqdm(range(start_epoch, 500)):
        
        stats = trainer.train(net, train_data, optimizer, loss_fn, device=device)

        wandb.log({'epoch' : i, 'loss' : stats['loss']}, step = stats['iterations'] + start_iterations)
        start_iterations = start_iterations + stats['iterations']
        
        
        validation_stats = trainer.valid(net, valid_data, loss_fn, device=device)
        
        wandb.log(validation_stats, step = start_iterations)
        
        scheduler.step(validation_stats['val_loss'])

        ### saving best model

        if validation_stats['val_loss'] < best_val_loss:
            best_val_loss = validation_stats['val_loss']
            torch.save(net, "./../weights/model_best.pth") # saved model

            validation_stats['epoch'] = i + 1
            json_object = json.dumps(validation_stats, indent = 4)
            with open("./../weights/model_best.json", "w") as outfile: # saved json containing metadata about best model
                json.dump(validation_stats, outfile)
        
        
        if (i + 1) % conf['EVAl_AFTER_EPOCHS'] == 0:

            torch.save({
                        'epoch': i,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': stats['loss'],
                        }, f"./../weights/model_{(i + 1)}.pth")
                    