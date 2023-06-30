import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
from functools import partial
import torch.nn as nn 
from datasets import load_dataset
from pytorch_lightning.loggers import TensorBoardLogger
from torchmultimodal.models.flava.model import flava_model_for_classification

import warnings
import sys 
warnings.filterwarnings("ignore")
num_gpus = int(sys.argv[1])
log_file = open("log.txt", "w")
with open("./vocabs/answers_textvqa_8k.txt") as f:
    VOCAB = [answer.strip() for answer in f.readlines()]
class TextVQADataset(Dataset):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __getitem__(self, index: int):
        return self.inner[index]

    def __len__(self):
        return len(self.inner)
class TextVQA(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                                       padding="max_length",
                                                       max_length=128)

    def setup(self, stage):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                                       padding="max_length",
                                                       max_length=128)
        with open("./vocabs/answers_textvqa_8k.txt") as f:
            self.vocab = [answer.strip() for answer in f.readlines()]

        self.answer_to_idx = {}
        for idx, entry in enumerate(self.vocab):
            self.answer_to_idx[entry] = idx

        self.dataset = load_dataset("textvqa")

        transform = partial(self._transform, self.tokenizer)
        self.dataset.set_transform(transform)

        train_dataset = TextVQADataset(self.dataset["train"])
        val_dataset = TextVQADataset(self.dataset["validation"])
        # val_dataset = TextVQADataset(self.dataset["validation"])

        self.train_dataset = DataLoader(train_dataset,
                                        batch_size=self.batch_size,
                                        num_workers=4*num_gpus,
                                        pin_memory=True,
                                        shuffle=True)
        self.val_dataset = DataLoader(val_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=4*num_gpus,
                                      pin_memory=True)

        self.model = FLAVAModel(num_classes=len(self.vocab))

    def _transform(self, tokenizer, input):
        batch = {}
        image_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize([224,224])])
        image = image_transform(input["image"][0].convert("RGB"))
        batch["image"] = [image]

        tokenized = tokenizer(input["question"],
                              return_tensors='pt',
                              padding="max_length",
                              max_length=128)
        batch.update(tokenized)

        ans_to_count = defaultdict(int)
        for ans in input["answers"][0]:
            ans_to_count[ans] += 1
        max_value = max(ans_to_count, key=ans_to_count.get)
        ans_idx = self.answer_to_idx.get(max_value, 0)
        batch["answers"] = torch.as_tensor([ans_idx])

        return batch

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

class FLAVAModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        # self.model = AutoModel.from_pretrained('/home/ubuntu/models--microsoft--git-base-coco/snapshots/a13141da42abd4a8cbf283601a8104265f537cee')
        # self.classifier = nn.Linear(768, num_classes)

        self.model = flava_model_for_classification(num_classes)

    def forward(self, text, image, labels):
        # logits = self.classifier(self.model(input_ids=text, pixel_values=image).last_hidden_state.mean(1))
        # loss = nn.CrossEntropyLoss()(logits, labels) 

        out = self.model(text=text, image=image, labels=labels)
        loss = out.loss
        logits = out.logits
        return {'loss': loss, 'logits': logits}

    def training_step(self, batch, batch_idx):
        out = self(text=batch["input_ids"],
                   image=batch["image"],
                   labels=batch["answers"])
        loss = out['loss']
        self.log('TRAIN_LOSS', loss, 
                 on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(text=batch["input_ids"],
                   image=batch["image"],
                   labels=batch["answers"])
        loss = out['loss']
        self.log('val_loss', loss, on_epoch=True)

        preds = torch.argmax(out['logits'], dim=-1)
        acc = ((preds == batch["answers"]) & (batch["answers"] != 0)).float().mean()
        if batch_idx == 0:
            tb = self.logger.experiment
            log_file.write(f'Epoch: {self.trainer.current_epoch}------------------------------------\n')
            for pred_idx, answer_idx in zip(preds, batch['answers']):
                log_file.write(f'{VOCAB[pred_idx]} vs {VOCAB[answer_idx]}\n')
                log_file.write('\n')

        self.log('val_acc', acc, on_epoch=True)   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-6)
        return optimizer
    
if __name__ == '__main__':
    textvqa = TextVQA(batch_size=32) 
    flava_model = FLAVAModel(num_classes=8001)
    logger = TensorBoardLogger(save_dir="tb-logger", name="debug")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filename="best_model",
    save_top_k=1,
    monitor="val_acc",
    mode="max"
    )
    trainer = pl.Trainer(logger=logger,
                        accelerator="gpu",
                        devices=num_gpus,
                        strategy="ddp_find_unused_parameters_true",
                        max_epochs=50,
                        callbacks=[checkpoint_callback])
    # trainer = pl.Trainer(logger=logger, accelerator="gpu", devices=1,  max_epochs=15, enable_checkpointing=False) #for debug


    trainer.fit(flava_model, datamodule=textvqa)