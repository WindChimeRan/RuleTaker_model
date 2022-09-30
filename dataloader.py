from dataset_reader import RuleTakerDataset
from torch.utils.data import DataLoader, default_collate

import pytorch_lightning as pl
import collections


class RuleTakerDataModule(pl.LightningDataModule):
    def __init__(self, train_path, dev_path, test_path, plm, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.plm = plm

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = RuleTakerDataset(self.plm, self.train_path)
            self.dev_dataset = RuleTakerDataset(self.plm, self.dev_path)
            # self.steps_per_epoch = len(self.train_dataset) // self.batch_size
        # Assign test dataset for use in dataloader(s)
        if stage == "validate":
            self.dev_dataset = RuleTakerDataset(self.plm, self.dev_path)

        if stage == "test":
            self.test_dataset = RuleTakerDataset(self.plm, self.test_path)

        # self.steps_per_epoch = len(self.train_dataset) // self.batch_size
        # total_training_steps = self.steps_per_epoch * N_EPOCHS
        # warmup_steps = total_training_steps // 5

    @staticmethod
    def custom_collate(batch):
        # print("custom_collate!!!!")
        elem = batch[0]
        # elem_type = type(elem)
        data = {
            key: default_collate([d[key] for d in batch])
            for key in elem
            if key != "metadata"
        }
        data["metadata"] = [d["metadata"] for d in batch]
        return data

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.custom_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.custom_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.custom_collate,
        )
