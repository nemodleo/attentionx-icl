import os
import fire

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AutoModelForSequenceClassification, AdamW

from iclx.soft_label_generator.datamodule.sst2 import SST2DataModule
from iclx.soft_label_generator.datamodule.sst5 import SST5DataModule
from iclx.soft_label_generator.datamodule.trec import TRECDataModule
from iclx.soft_label_generator.datamodule.ag_news import AGNewsDataModule
from iclx.soft_label_generator.datamodule.yelp import YelpDataModule
from iclx.soft_label_generator.datamodule.mnli import MNLIDataModule
from iclx.soft_label_generator.datamodule.qnli import QNLIDataModule


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
    model_name_or_path: str = "bert-base-uncased",
    dataset: str = "sst2",
    max_epochs: int = 100,
    n_gpus: int = 8,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_token_len: int = 512,
):
    if dataset == "sst2":
        data_module = SST2DataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset == "sst5":
        data_module = SST5DataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset == "trec":
        data_module = TRECDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset == "ag_news":
        data_module = AGNewsDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset == "yelp":
        data_module = YelpDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset == "mnli":
        data_module = MNLIDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset == "qnli":
        data_module = QNLIDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

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

    # Start training
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        devices=n_gpus,
        accelerator="cuda",
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    fire.Fire(train)