from dataset_reader import RuleTakerDataset
from torch.utils.data import DataLoader, default_collate

import pytorch_lightning as pl
import collections


class RuleTakerDataModule(pl.LightningDataModule):
    def __init__(self, train_path, dev_path, test_path, encoder_name, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.pretrained_model = encoder_name
        # self.train_dataset = RuleTakerDataset(self.pretrained_model, self.train_path)
        # self.test_dataset = RuleTakerDataset(self.pretrained_model, self.test_path)

    def setup(self, stage=None):
        self.train_dataset = RuleTakerDataset(self.pretrained_model, self.train_path)
        self.dev_dataset = RuleTakerDataset(self.pretrained_model, self.dev_path)
        self.test_dataset = RuleTakerDataset(self.pretrained_model, self.test_path)

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
        # if isinstance(elem, collections.abc.Mapping):
        #     try:
        #         return elem_type({key: default_collate([d[key] for d in batch]) for key in elem if key != "metadata"})
        #     except TypeError:
        #         # The mapping type may not support `__init__(iterable)`.
        #         return {key: default_collate([d[key] for d in batch]) for key in elem}

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
            self.test_dataset,
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
