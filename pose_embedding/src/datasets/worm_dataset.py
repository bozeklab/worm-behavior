import torch
import torchvision
from torch.utils.data import Dataset


class WormDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        step: int,
        transform: torchvision.transforms.Compose,
        view_aug: torchvision.transforms.Compose | None = None
    ):
        # load images
        self.dataset_path = dataset_path
        self.step = step
        self.transform = transform
        self.view_aug = view_aug

        self.img_dataset = torch.load(self.dataset_path)
        self.img_dataset = self.img_dataset[:: self.step, None, :, :]

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img = self.img_dataset[idx]
        second_view = img.detach().clone()

        if self.view_aug is not None:
            second_view = self.view_aug(second_view)
        else:
            second_view = self.transform(second_view)

        if self.view_aug is not None:
            img = self.view_aug(img)
        else:
            img = self.transform(img)

        return img, second_view
