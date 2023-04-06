from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class DataManager:
    
    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size
        self._download()
        self._load()
        
    def get_dataloader(self, train: bool = True) -> DataLoader:
        
        if train:
            return self._ds_loader_train
        else:
            return self._ds_loader_test
    
    def _download(self):
        
        # download training dataset from open datasets.
        self._ds_train = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        # download test dataset from open datasets.
        self._ds_test = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )
    
    def _load(self):
        
        self._ds_loader_train = DataLoader(self._ds_train, batch_size=self._batch_size)
        self._ds_loader_test = DataLoader(self._ds_test, batch_size=self._batch_size)