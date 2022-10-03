import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from typing import Any, Tuple, Iterable

import torch.nn as nn
import torch


# MODEL_NAME = "roberta-base"


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
        self.metrics = torchmetrics.Accuracy()

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
            # TODO: acc, metadata
            is_correct = output_dic["answer_index"] == label
            output_dic["is_correct"] = is_correct

        return loss, output_dic

    def training_step(self, batch, batch_idx):
        # data, metadata = batch

        input_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.metrics(outputs["answer_index"], label)
        acc = self.metrics
        self.log("performance", {"loss": loss, "acc": acc})
        # self.log(f"train_loss={loss}\tAcc={acc}", loss, prog_bar=True, logger=True, on_epoch=True)
        return {"loss": loss, "predictions": outputs["answer_index"], "label": label}

    def test_step(self, batch, batch_idx):
        return self.val_test_step("test", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.val_test_step("val", batch, batch_idx)

    def val_test_step(self, split, batch, batch_idx):

        input_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.metrics(outputs["answer_index"], label)
        acc = self.metrics
        self.log("performance", {"loss": loss, "acc": acc})
        # self.log(f"{split}_loss={loss}\tAcc={acc}", prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "predictions": outputs["answer_index"], "label": label}

    # def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
    #     return self.val_test_epoch_end("val", outputs)

    # def test_epoch_end(self, outputs: Iterable[Any]) -> None:
    #     return self.val_test_epoch_end("test", outputs)

    # def training_epoch_end(self, outputs):

    #     labels = []
    #     predictions = []
    #     for output in outputs:
    #         for out_labels in output["label"].detach().cpu():
    #             labels.append(out_labels)
    #         for out_predictions in output["predictions"].detach().cpu():
    #             predictions.append(out_predictions)

    #     labels = torch.stack(labels).int()
    #     predictions = torch.stack(predictions)

    # def val_test_epoch_end(self, split: str, outputs: Iterable[Any]) -> None:
    #     results = []

    def configure_optimizers(self):
        pass
