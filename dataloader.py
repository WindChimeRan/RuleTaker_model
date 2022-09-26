from dataset_reader import RuleTakerDataset

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class RuleTakerDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, pretrained_model, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.train_path = train_path
        self.test_path = test_path
        self.pretrained_model = pretrained_model

    def setup(self, stage=None):
        self.train_dataset = RuleTakerDataset(self.pretrained_model, self.train_path)
        self.test_dataset = RuleTakerDataset(self.pretrained_model, self.test_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
