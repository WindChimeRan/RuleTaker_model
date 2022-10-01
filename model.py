import pytorch_lightning as pl

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

    def test_step(self, batch, batch_idx):
        return self.val_test_step("test", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.val_test_step("val", batch, batch_idx)

    def val_test_step(self, split, batch, batch_idx):

        input_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]
        loss, outputs = self(input_ids, attention_mask, label)
        self.log(f"loss_{split}", loss, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "predictions": outputs["answer_index"], "label": label}

    def validation_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("val", outputs)

    def test_epoch_end(self, outputs: Iterable[Any]) -> None:
        return self.val_test_epoch_end("test", outputs)

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

    def val_test_epoch_end(self, split: str, outputs: Iterable[Any]) -> None:
        results = []

        for out in outputs:
            # if self.dataset == "entailmentbank":
            #     for proof_pred, score, proof in zip(*out):
            #         results.append(
            #             {
            #                 "proof_pred": proof_pred,
            #                 "score": score,
            #                 "hypothesis": proof.hypothesis,
            #                 "context": proof.context,
            #                 "proof_gt": proof.proof_text,
            #             }
            #         )
            # else:
            for proof_pred, score, proof, answer, depth, all_proofs in zip(*out):
                results.append(
                    {
                        "answer": answer,
                        "depth": depth,
                        "all_proofs": all_proofs,
                        "proof_pred": proof_pred,
                        "score": score,
                        "hypothesis": proof.hypothesis,
                        "context": proof.context,
                        "proof_gt": proof.proof_text,
                    }
                )

        assert self.trainer is not None
        if self.logger is not None and self.trainer.log_dir is not None:
            json_path = os.path.join(self.trainer.log_dir, f"results_{split}.json")
            json.dump(results, open(json_path, "wt"))
            if self.dataset == "entailmentbank":
                tsv_path = os.path.join(self.trainer.log_dir, f"results_{split}.tsv")
                with open(tsv_path, "wt") as oup:
                    for r in results:
                        proof = r["proof_pred"].strip()
                        if not proof.endswith(";"):
                            proof += ";"
                        oup.write(f"$proof$ = {proof}\n")
                print(f"Validation results saved to {json_path} and {tsv_path}")
            else:
                print(f"Validation results saved to {json_path}")

        # if self.dataset == "entailmentbank" and results[0]["proof_gt"] != "":
        #     em, f1 = evaluate_entailmentbank(results, eval_intermediates=False)
        #     for k, v in em.items():
        #         self.log(f"ExactMatch_{k}_{split}", v, on_step=False, on_epoch=True)
        #     for k, v in f1.items():
        #         self.log(f"F1_{k}_{split}", v, on_step=False, on_epoch=True)

        # elif self.dataset == "ruletaker":
        answer_accuracies, proof_accuracies = evaluate_ruletaker(results)
        for k in answer_accuracies.keys():
            self.log(
                f"Accuracy_answer_{k}_{split}",
                answer_accuracies[k],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"Accuracy_proof_{k}_{split}",
                proof_accuracies[k],
                on_step=False,
                on_epoch=True,
            )

        # TODO: add ACC

        # for i, name in enumerate(LABEL_COLUMNS):
        #     class_roc_auc = auroc(predictions[:, i], labels[:, i])
        #     self.logger.experiment.add_scalar(
        #         f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch
        #     )

    def configure_optimizers(self):
        pass

    def calculate_score(self):
        pass

    #     optimizer = AdamW(self.parameters(), lr=1e-5)

    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.n_warmup_steps,
    #         num_training_steps=self.n_training_steps,
    #     )

    #     return dict(
    #         optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
    #     )
