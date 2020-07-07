import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import torch.utils.data as data
from PIL import Image

class ADV_MNIST(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        class_folder = os.listdir(root)
        self.data = []
        self.labels = []
        for f in class_folder:
            label = int(f)
            images = os.listdir(os.path.join(root,f))
            for fi in images:
                fp = open(os.path.join(root,f,fi),'rb')
                image = Image.open(fp)
                image = np.array(image)
                self.data.append(image)
                fp.close()
                self.labels.append(label)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)




