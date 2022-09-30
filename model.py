import pytorch_lightning as pl

# from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

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
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs["answer_index"], "label": label}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # data, metadata = batch

        input_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["label"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        # TODO: add ACC

        # for i, name in enumerate(LABEL_COLUMNS):
        #     class_roc_auc = auroc(predictions[:, i], labels[:, i])
        #     self.logger.experiment.add_scalar(
        #         f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
        #     )

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
