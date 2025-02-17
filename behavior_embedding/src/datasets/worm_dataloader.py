import lightning.pytorch as pl
import numpy as np
import torch
from torchvision.transforms import v2

from datasets.worm_dataset import WormDataset
from sampler.sequential_batch_sampler import SequentialBatchSampler


class WormDataLoader(pl.LightningDataModule):

    def __init__(
        self,
        train_dataset_path: str,
        val_dataset_path: str,
        test_dataset_path: str,
        dataset_mean: list[float],
        dataset_std: list[float],
        seq_length: int,
        batch_size: int = 512,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        img_size: int = 64,
        persistent_workers: bool = False,
        step: int = 1,
        enc_rotation: bool = True,
        train_rotation_path: str = None,
        val_rotation_path: str = None,
        test_rotation_path: str = None,
        train_meta_data_path: str = None,
        val_meta_data_path: str = None,
        test_meta_data_path: str = None,
        id_stepping: str = "multi",
        train_tierpsy_feat_path=None,
        val_tierpsy_feat_path=None,
        test_tierpsy_feat_path=None,
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.test_dataset_path = test_dataset_path
        self.train_rotation_path = train_rotation_path
        self.val_rotation_path = val_rotation_path
        self.test_rotation_path = test_rotation_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.img_size = img_size
        self.persistent_workers = persistent_workers
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.seq_length = seq_length
        self.step = step
        self.enc_rotation = enc_rotation
        self.train_meta_data_path = train_meta_data_path
        self.val_meta_data_path = val_meta_data_path
        self.test_meta_data_path = test_meta_data_path
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.id_stepping = id_stepping
        self.train_tierpsy_feat_path = train_tierpsy_feat_path
        self.val_tierpsy_feat_path = val_tierpsy_feat_path
        self.test_tierpsy_feat_path = test_tierpsy_feat_path

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize((self.img_size, self.img_size), antialias=False),
                v2.Lambda(lambda x: x.repeat(1, 3, 1, 1)),
                v2.ToDtype(torch.float32, scale=True),  # replacement for toTensor
                v2.Normalize(self.dataset_mean, self.dataset_std),
            ]
        )

        self.train_transform = v2.Compose(
            [
                # START Transforms
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=False),
                v2.Resize((self.img_size, self.img_size), antialias=False),
                # Augmentations
                v2.RandomChoice(
                    [
                        v2.CenterCrop(
                            size=int(
                                self.img_size
                                * ((0.95 - 0.85) * np.random.random_sample() + 0.85)
                            )
                        ),
                        v2.RandomZoomOut(fill=127, side_range=(1.0, 1.8), p=1.0),
                    ],
                    p=[0.5, 0.5],
                ),
                v2.RandomApply(
                    [
                        v2.GaussianBlur(kernel_size=3),
                    ],
                    p=0.5,
                ),
                v2.RandomApply(
                    [v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5])], p=0.5
                ),
                v2.RandomInvert(p=0.7),
                # END Transforms
                v2.Lambda(lambda x: x.repeat(1, 3, 1, 1)),
                v2.Resize((self.img_size, self.img_size)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(self.dataset_mean, self.dataset_std),
            ]
        )

    def setup(self, stage: str):
        if stage == "fit":
            # load images
            self.train_dataset = WormDataset(
                self.train_dataset_path,
                self.train_rotation_path,
                self.step,
                self.train_transform,
                self.train_tierpsy_feat_path,
            )
            print(f"Train dataset len: {len(self.train_dataset)}")

            self.val_dataset = WormDataset(
                self.val_dataset_path,
                self.val_rotation_path,
                self.step,
                self.transform,
                self.val_tierpsy_feat_path,
            )
            print(f"Val dataset len: {len(self.val_dataset)}")

        elif stage == "test":
            # load images
            self.test_dataset = WormDataset(
                self.test_dataset_path,
                self.test_rotation_path,
                self.step,
                self.transform,
                self.test_tierpsy_feat_path,
            )
            print(f"Test dataset len: {len(self.test_dataset)}")

        elif stage == "predict":
            # load images
            self.test_dataset = WormDataset(
                self.test_dataset_path,
                self.test_rotation_path,
                self.step,
                self.transform,
                self.test_tierpsy_feat_path,
            )
            print(f"Test dataset len: {len(self.test_dataset)}")

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=SequentialBatchSampler(
                self.train_dataset,
                self.train_meta_data_path,
                seq_length=self.seq_length,
                shuffle=self.shuffle,
                step=self.step,
                stage="train",
                id_stepping=self.id_stepping,
            ),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            sampler=SequentialBatchSampler(
                self.val_dataset,
                self.val_meta_data_path,
                seq_length=self.seq_length,
                step=self.step,
                stage="val",
                id_stepping=self.id_stepping,
            ),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            sampler=SequentialBatchSampler(
                self.test_dataset,
                self.test_meta_data_path,
                seq_length=self.seq_length,
                step=self.step,
                stage="test",
                id_stepping=self.id_stepping,
            ),
            batch_size=1,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return test_dataloader

    def predict_dataloader(self):
        predict_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            sampler=SequentialBatchSampler(
                self.test_dataset,
                self.test_meta_data_path,
                seq_length=self.seq_length,
                step=self.step,
                stage="predict",
                id_stepping=self.id_stepping,
            ),
            batch_size=1,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return predict_dataloader
