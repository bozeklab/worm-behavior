import lightning.pytorch as pl
import torch
from torchvision.transforms import v2
import numpy as np

from datasets.worm_dataset import WormDataset


class WormDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset_path,
        val_dataset_path,
        test_dataset_path,
        dataset_mean,
        dataset_std,
        batch_size: int = 512,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        img_size: int = 64,
        persistent_workers: bool = False,
        step: int = 1,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.img_size = img_size
        self.persistent_workers = persistent_workers
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.step = step

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize((self.img_size, self.img_size), antialias=False),
                v2.Lambda(lambda x: x.repeat(3, 1, 1)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(self.dataset_mean, self.dataset_std),
            ]
        )

        self.train_view_aug = v2.Compose(
            [
                # START Transforms
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=False),
                v2.Resize((self.img_size, self.img_size), antialias=False),

                # Augmentations
                v2.RandomChoice(
                    [
                        v2.CenterCrop(size=int(self.img_size * ((0.95 - 0.85) * np.random.random_sample() + 0.85))),
                        v2.RandomZoomOut(fill=127, side_range=(1.0, 1.8), p=1.0),
                    ],
                    p=[0.5, 0.5],
                ),
                v2.RandomApply(
                    [
                        v2.RandomAffine(translate=(0.1, 0.1), degrees=0, fill=127),
                    ],
                    p=0.7
                ),
                v2.RandomChoice([
                    v2.RandomHorizontalFlip(p=1.0),
                    v2.RandomVerticalFlip(p=1.0),
                    v2.RandomRotation(degrees=90, fill=127), #127
                ], [0.33, 0.33, 0.33]),
                v2.RandomApply(
                    [
                        v2.GaussianBlur(kernel_size=3),
                    ],
                    p=0.5
                ),
                v2.RandomApply(
                    [

                        v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5])
                    ],
                    p=0.5
                ),
                v2.RandomInvert(p=0.7),

                # END Transforms
                v2.Lambda(lambda x: x.repeat(3, 1, 1)),
                v2.Resize((self.img_size, self.img_size)),
                v2.ToDtype(torch.float32, scale=True),  # replacement for toTensor
                v2.RandomApply(
                    [
                        v2.GaussianNoise(),
                    ],
                    p=0.5
                ),
                v2.RandomGrayscale(p=0.2),
                v2.Normalize(self.dataset_mean, self.dataset_std),
            ]
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = WormDataset(
                self.train_dataset_path,
                self.step,
                self.transform,
                self.train_view_aug
            )
            print(f"Train dataset len: {len(self.train_dataset)}")

            self.val_dataset = WormDataset(
                self.val_dataset_path, self.step, self.transform, self.train_view_aug
            )
            print(f"Val dataset len: {len(self.val_dataset)}")

        elif stage == "test":
            # load images
            self.test_dataset = WormDataset(
                self.test_dataset_path,
                self.step,
                self.transform,
                self.train_view_aug
            )
            print(f"Test dataset len: {len(self.test_dataset)}")

        elif stage == "predict":
            # load images
            self.test_dataset = WormDataset(
                self.test_dataset_path,
                self.step,
                self.transform,
                self.train_view_aug
            )
            print(f"Test dataset len: {len(self.test_dataset)}")

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return test_dataloader

    def predict_dataloader(self):
        predict_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return predict_dataloader

