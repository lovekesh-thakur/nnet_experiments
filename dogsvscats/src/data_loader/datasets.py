from torch.utils.data import Dataset, DataLoader
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DogsVsCatsDataset(Dataset):
    """
    libsum alpha beta gamma
    """
    def __init__(self, image_file, transforms = None):
        super().__init__()
        self.image_file = image_file
        self.transforms = transforms
        self.images_filepath = [i.strip() for i in open(self.image_file, 'r')]
    
    def __len__(self):
        return len(self.images_filepath)

    def __getitem__(self, idx):
        image_filepath = self.images_filepath[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if "cat" in image_filepath.split(os.sep)[-1]:
            label = 1
        else:
            label = 0
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        return image, label


if __name__ == '__main__':
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
    dataset = DogsVsCatsDataset("/home/lovekesh/Documents/nnet_experiments/dogsvscats/dataset/train/images.txt", 
                                transforms=train_transform)
    data = DataLoader(dataset=dataset, batch_size=16, num_workers=4, shuffle=True)
    for x, y in data:
        print(x.shape)
        print(y.shape)

       
    
