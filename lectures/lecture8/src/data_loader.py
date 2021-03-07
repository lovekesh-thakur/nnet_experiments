from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
#import pytorch_lightning as pl

class TorchvisionDataLoader():
    """
    Data loader using lightning module
    """
    def __init__(self, args):
        super().__init__()
        self.kwargs = args
        self.dataset_class = getattr(self.kwargs, 'dataset')
        self.data_dir = getattr(self.kwargs, 'data_dir')
        self.batch_size = getattr(self.kwargs, 'batch_size')
        self.DataLoader = getattr(datasets, self.dataset_class)
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            lambda x: x>0,
            lambda x: x.view(28*28, -1),
            lambda x: x.float()])
    
    def prepare_data(self):
        ## load class from datasets
        
        # download dataset
        self.DataLoader(self.data_dir, download = True, train = True)
        self.DataLoader(self.data_dir, download = True, train = False)
        return self
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_full = self.DataLoader(self.data_dir, train = True, transform = self.transform)
            self.train_data, self.val_data = random_split(self.data_full, [55000, 5000])
        if stage == 'test' :
            self.test_data = self.DataLoader(self.data_dir, train = False)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size)
    
            





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'MNIST', help="name of dataset class to load from Torchvision")
    parser.add_argument('--batch_size', type=int, default = 16, help="Batch Size")
    parser.add_argument('--data_dir', type=str, default = './../../lecture7/data/', help="folder location")
 
    args = parser.parse_args()

    
    data_loader = TorchvisionDataLoader(args)
    data_loader.prepare_data()
    data_loader.setup(stage = 'fit')
    data = data_loader.train_dataloader()
    a, b = next(iter(data))