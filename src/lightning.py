import os
from abc import ABC

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from typing import Any, Dict
from argparse import Namespace
from src import data, model, tokenizer
from torch.utils.data import DataLoader


class LightningSequence2Sequence(pl.LightningModule, ABC):

    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams

        source_language_tokenizer, target_language_tokenizer = tokenizer.load_seq2seq_tokenizers(
            directory=self.hparams.directory)

        self.sequence2sequence_preparer = tokenizer.Sequence2SequencePreparer(
            source_language_tokenizer=source_language_tokenizer,
            target_language_tokenizer=target_language_tokenizer)

        if self.hparams.use_attention:
            self.model = model.Sequence2SequenceWithAttentionModel(config=self.hparams)
        else:
            self.model = model.Sequence2SequenceModel(config=self.hparams)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.pad_index)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(params=self.parameters(),
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        return optimizer

    def get_dataloader(self, data_type: str) -> DataLoader:

        source_data = data.load_file(
            os.path.join(self.hparams.directory, f'{data_type}_en.txt'))
        target_data = data.load_file(
            os.path.join(self.hparams.directory, f'{data_type}_ru.txt'))

        dataset = data.Sequence2SequenceDataset(source_language_data=source_data,
                                                target_language_data=target_data)

        loader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.sequence2sequence_preparer.collate
        )

        return loader

    def train_dataloader(self):
        loader = self.get_dataloader(data_type='train')

        return loader

    def val_dataloader(self):
        loader = self.get_dataloader(data_type='valid')

        return loader

    def forward(self,
                source_sequence: torch.Tensor,
                target_sequence: torch.Tensor) -> torch.Tensor:

        logits = self.model(source_sequence, target_sequence)

        return logits

    def training_step(self, batch: Any, batch_idx: int) -> Dict:

        source_ids, target_ids, target_criterion_ids = batch

        logits = self.forward(source_ids, target_ids)

        prediction, target = logits.reshape(-1, logits.size(-1)), target_criterion_ids.contiguous().view(-1)

        loss = self.criterion(prediction, target)

        log = {
            'train_loss': loss.item(),
            'train_perplexity': np.exp(loss.item())
        }

        return {'loss': loss, 'log': log}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict:

        source_ids, target_ids, target_criterion_ids = batch

        logits = self.forward(source_ids, target_ids)

        prediction, target = logits.reshape(-1, logits.size(-1)), target_criterion_ids.contiguous().view(-1)

        loss = self.criterion(prediction, target)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        log = {
            'val_loss': avg_loss,
            'val_perplexity': np.exp(avg_loss.item())
        }

        return {'val_loss': avg_loss, 'log': log}
