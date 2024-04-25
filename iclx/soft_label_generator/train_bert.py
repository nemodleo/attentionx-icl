import os
import fire

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from transformers import AutoModelForSequenceClassification, AdamW

from iclx.soft_label_generator.datamodule.sst2 import SST2DataModule
from iclx.soft_label_generator.datamodule.sst5 import SST5DataModule
from iclx.soft_label_generator.datamodule.trec import TRECDataModule
from iclx.soft_label_generator.datamodule.ag_news import AGNewsDataModule
from iclx.soft_label_generator.datamodule.yelp import YelpDataModule
from iclx.soft_label_generator.datamodule.mnli import MNLIDataModule
from iclx.soft_label_generator.datamodule.qnli import QNLIDataModule
from iclx.soft_label_generator.datamodule.mnist_text import MNISTTextDataModule


def initialize_data_module(dataset, model_name_or_path, batch_size, max_token_len, sampling_rate):
    data_modules = {
        "sst2": SST2DataModule,
        "sst5": SST5DataModule,
        "trec": TRECDataModule,
        "ag_news": AGNewsDataModule,
        "yelp": YelpDataModule,
        "mnli": MNLIDataModule,
        "qnli": QNLIDataModule,
    }
    if dataset in data_modules:
        return data_modules[dataset](
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
            sampling_rate=sampling_rate,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


class BERTTrainingModule(pl.LightningModule):
    def __init__(self, model_name_or_path: str, num_labels: int, lr: float):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
        )
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        outputs = self(**inputs)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        outputs = self(**inputs)
        val_loss = outputs.loss
        self.log('val_loss', val_loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
    
    def on_save_checkpoint(self, checkpoint):
        self.model.save_pretrained("model")


def train(
    dataset: str = "sst2",
    lr: float = 2e-5,
    batch_size: int = 64,
    sampling_rate: float = 1.0,
    max_token_len: int = 512,
    n_gpus: int = 1,
    max_epochs: int = 100,
    model_name_or_path: str = "bert-base-uncased",
    device: str = "cuda"
):
    data_module = initialize_data_module(
        dataset,
        model_name_or_path,
        batch_size,
        max_token_len,
        sampling_rate,
    )

    data_module.setup()

    # Load model
    model = BERTTrainingModule(
        model_name_or_path=model_name_or_path,
        num_labels=data_module.num_labels(),
        lr=lr,
    )

    # Set checkpoint callback
    checkpoint_dir = f"checkpoints/{dataset}/{model_name_or_path}__lr_{lr}__bs_{batch_size}__n_gpus_{n_gpus}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
        dirpath=checkpoint_dir,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min"
    )

    # Start training
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        devices=n_gpus,
        accelerator=device,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    fire.Fire(train)
