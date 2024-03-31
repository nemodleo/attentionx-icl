import fire
from datasets import load_dataset

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW


class SST2DataModule(pl.LightningDataModule):
    def __init__(self, model_name_or_path: str, batch_size: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = load_dataset("glue", "sst2")
        self.train_dataset, self.val_dataset = self._tokenize(dataset['train']), self._tokenize(dataset['validation'])

    def _tokenize(self, dataset):
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples['sentence'],
                padding=True,
                truncation=True,
                max_length=512,
            ),
            batched=True,
        )
        return tokenized_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def num_labels(self):
        return 2


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
    max_epochs: int = 10,
    n_gpus: int = 8,
    batch_size: int = 32,
    lr: float = 2e-5,
):
    # Load data
    data_module = SST2DataModule(
        model_name_or_path=model_name_or_path,
        batch_size=batch_size,
    )

    # Load model
    model = BERTTrainingModule(
        model_name_or_path=model_name_or_path,
        num_labels=data_module.num_labels(),
        lr=lr,
    )

    # Start training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=n_gpus,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    fire.Fire(train)
