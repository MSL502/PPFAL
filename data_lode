import os
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_syn_dataloader(path, batch_size=64):
    train_loader = DataLoader(train_Dataset(path), batch_size=batch_size, shuffle=True, drop_last=False)
    return train_loader


def test_dataloader(path, batch_size=1):
    test_loader = DataLoader(test_Dataset(path), batch_size=batch_size, shuffle=False, drop_last=False)
    return test_loader


class train_Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.low_image_list = os.listdir(os.path.join(image_dir, 'low'))
        self.high_image_list = os.listdir(os.path.join(image_dir, 'high'))
        self._check_image(self.low_image_list)
        self.low_image_list.sort()
        self.transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )

    def __len__(self):
        return len(self.low_image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'low', self.low_image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'high', self.low_image_list[idx].split('/')[-1].split('.')[0]+'.png'))
        image, label = self.transform(image, label)
        name = self.low_image_list[idx]
        return image, label, name

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'bmp']:
                raise ValueError


class test_Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.low_image_list = os.listdir(os.path.join(image_dir, 'low'))
        self.high_image_list = os.listdir(os.path.join(image_dir, 'high'))
        self._check_image(self.low_image_list)
        self.low_image_list.sort()

    def __len__(self):
        return len(self.low_image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'low', self.low_image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'high', self.low_image_list[idx].split('/')[-1].split('.')[0]+'.png'))
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        name = self.low_image_list[idx]
        return image, label, name

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'bmp']:
                raise ValueError

