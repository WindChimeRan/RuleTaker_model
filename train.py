from model import RuleTakerModel
from dataloader import RuleTakerDataModule


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


import pytorch_lightning as pl


RANDOM_SEED = 71


pl.seed_everything(RANDOM_SEED)

backbone = "roberta-large"
dataset_dir = (
    "/data/hzz5361/raw_data/rule-reasoning-dataset-V2020.2.5.0/original/depth-3/"
)


N_EPOCHS = 3
batch_size = 8
train_data_path = dataset_dir + "train.jsonl"
validation_data_path = dataset_dir + "dev.jsonl"
test_data_path = dataset_dir + "test.jsonl"

data_module = RuleTakerDataModule(
    train_path=train_data_path,
    dev_path=validation_data_path,
    test_path=test_data_path,
    encoder_name=backbone,
    batch_size=batch_size,
)

data_module.setup()

steps_per_epoch = len(data_module.train_dataset) // batch_size
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5


checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

logger = TensorBoardLogger("lightning_logs", name="leapofthought")

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

model = RuleTakerModel(
    plm=backbone,
    n_classes=2,
    n_training_steps=total_training_steps,
    n_warmup_steps=warmup_steps,
)

trainer = pl.Trainer(
    logger=logger,
    # checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback, checkpoint_callback],
    max_epochs=N_EPOCHS,
    accelerator="gpu",
    gpus=[6, 7],
    # gpus=[7],
    enable_progress_bar=True
    # num_sanity_val_steps=0,
)

if __name__ == "__main__":
    trainer.fit(model, data_module)
