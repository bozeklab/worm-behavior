import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class WormDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        rotation_path: str,
        step: int,
        transform: torchvision.transforms.Compose,
        tierpsy_feat_path: str = None,
    ):

        # load images
        self.dataset_path = dataset_path
        self.step = step
        self.rotation_path = rotation_path
        self.transform = transform
        self.tierpsy_feat_path = tierpsy_feat_path

        self.img_dataset = torch.load(self.dataset_path)
        self.img_dataset = self.img_dataset[:: self.step, None, :, :]

        # load rotations
        if self.rotation_path is not None:
            self.rotations = self.load_rotation_list(self.rotation_path)
            # Stepping
            self.rotations = self.rotations[:: self.step]

    @staticmethod
    def load_rotation_list(path):
        with open(path, "r") as f:
            rotations = f.readlines()
            # convert rotations from 0-180 to 0-1
            rotations = np.array(rotations, dtype=np.float32)
            rotations = rotations / 180.0
            return torch.FloatTensor(rotations)

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img = self.img_dataset[idx]
        img = self.transform(img)
        if self.rotation_path is not None:
            rotation = self.rotations[idx]

            return img, rotation
        else:
            return img
