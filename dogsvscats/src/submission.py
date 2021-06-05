import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from tqdm import tqdm
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


valid_transform = A.Compose(
                            [
                            A.Resize(300, 300),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2(),
                            ]
                            )

model = torch.load("./../weights/model_best.pth").eval().to(device)

test_file = "/home/lovekesh/Developer/nnet_experiments/dogsvscats/data/test_images.txt"

f = open('submit.txt', 'w')
f.write("id,label" + "\n")
images = [i.strip() for i in open(test_file, 'r')]

for img_p in tqdm(images):
    img_p = img_p.strip()
    image = cv2.imread(img_p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    
    image = valid_transform(image=image)["image"]
    image = torch.unsqueeze(image, 0).to(device)
    out = model(image)
    proba = round(torch.softmax(out, dim = 1).cpu().data[0][0].item(), 4)
    if proba > 0.95:
        proba = 0.95
    if proba < 0.05:
        proba = 0.05
    image_id = img_p.split(os.sep)[-1].split('.')[0]
    msg = ",".join([str(image_id), str(proba)])
    f.write(msg + "\n")

    
f.close()