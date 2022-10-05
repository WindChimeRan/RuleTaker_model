import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoModel


from typing import Any, Tuple, Iterable

import torch.nn as nn


class RuleTakerModel(pl.LightningModule):
    def __init__(
        self, plm: str, n_classes: int, n_training_steps=None, n_warmup_steps=None,
    ):
        super().__init__()
        self.plm = plm
        self.encoder = AutoModel.from_pretrained(plm, return_dict=True)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, n_classes)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.CrossEntropyLoss()

        self.train_metrics = torchmetrics.Accuracy()
        self.val_metrics = torchmetrics.Accuracy()
        # self.train_metrics = torchmetrics.Accuracy()
        # self.val_metrics = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask, label=None):
        output = self.encoder(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        label_logits = self.dropout(output)
        output_dic = {}
        loss = 0

        output_dic["label_logits"] = label_logits
        output_dic["label_probs"] = nn.functional.softmax(label_logits, dim=1)
        output_dic["answer_index"] = label_logits.argmax(1)

        if label is not None:
            loss = self.criterion(label_logits, label)
            is_correct = output_dic["answer_index"] == label
            output_dic["is_correct"] = is_correct

        return loss, output_dic

    # def log_metrics(self, split, pred, label) -> None:
    #     self.metrics[split](pred, label)
    #     # self.metrics[split](pred, label)
    #     self.log(f"Acc_{split}", self.metrics[split], on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):

        input_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.log("loss_train", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "predictions": outputs["answer_index"], "label": label}

    def training_step_end(self, step_output):
        split = "train"
        self.train_metrics(step_output["predictions"], step_output["label"])
        self.log(f"Acc_{split}", self.train_metrics, on_step=False, on_epoch=True)
        # self.log_metrics("train", step_output["predictions"], step_output["label"])

    def validation_step_end(self, step_output):
        # self.log_metrics("val", step_output["predictions"], step_output["label"])
        split = "val"
        self.val_metrics(step_output["predictions"], step_output["label"])
        self.log(f"Acc_{split}", self.val_metrics, on_step=False, on_epoch=True)

    # def test_step(self, batch, batch_idx):
    #     return self.val_test_step("test", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.val_test_step("val", batch, batch_idx)

    def val_test_step(self, split, batch, batch_idx):

        input_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        # self.metrics["val"](outputs["answer_index"], label)
        # self.log_metrics("val", outputs["answer_index"], label)
        # acc = self.metrics
        self.log("loss_val", loss, on_step=True, on_epoch=True)

        # self.log("performance", {"loss": loss, "acc": acc})
        # self.log(f"{split}_loss={loss}\tAcc={acc}", prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "predictions": outputs["answer_index"], "label": label}

    def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("val", outputs)

    # def test_epoch_end(self, outputs: Iterable[Any]) -> None:
    #     return self.val_test_epoch_end("test", outputs)

    def training_epoch_end(self, outputs):
        self.train_metrics.reset()

    def val_test_epoch_end(self, split: str, outputs: Iterable[Any]) -> None:
        val = self.val_metrics.compute()
        self.log("val_acc_epoch", val)
        self.val_metrics.reset()
        print("val acc ", val)

    def configure_optimizers(self):
        pass
