import os
from abc import ABC
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from typing import List, Any, Dict
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
            os.path.join(self.hparams.directory, f'{data_type}_{self.hparams.source_data}.txt'))
        target_data = data.load_file(
            os.path.join(self.hparams.directory, f'{data_type}_{self.hparams.target_data}.txt'))

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

    @staticmethod
    def add_model_specific_args(parent_parser):

        parser = ArgumentParser(parents=[parent_parser])

        # model
        parser.add_argument('--model_dim', type=int, default=256)
        parser.add_argument('--feed_forward_dim', type=int, default=3072)
        parser.add_argument('--num_layers', type=int, default=12)
        parser.add_argument('--response_segment_index', type=int, default=1)
        parser.add_argument('--query_segment_index', type=int, default=2)
        parser.add_argument('--context_segment_index', type=int, default=3)
        parser.add_argument('--weight_tying', action='store_true')
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--initializer_range', type=float, default=0.02)

        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.)

        return parser

    def forward(self,
                source_sequence: torch.Tensor,
                target_sequence: torch.Tensor) -> torch.Tensor:

        logits = self.model(source_sequence, target_sequence)

        return logits

    def training_step(self, batch: Any, batch_idx: int) -> Dict:

        source_ids, target_ids, target_criterion_ids = batch

        logits = self.forward(source_ids, target_ids)

        prediction, target = logits.reshape(-1, logits.size(-1)), target_criterion_ids.view(-1)

        loss = self.criterion(prediction, target)

        log = {
            'train_loss': loss.item(),
            'train_perplexity': np.exp(loss.item())
        }

        return {'loss': loss, 'log': log}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict:

        source_ids, target_ids, target_criterion_ids = batch

        logits = self.forward(source_ids, target_ids)

        prediction, target = logits.reshape(-1, logits.size(-1)), target_criterion_ids.view(-1)

        loss = self.criterion(prediction, target)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        log = {
            'val_loss': avg_loss,
            'val_perplexity': np.exp(avg_loss.item())
        }

        return {'val_loss': avg_loss, 'log': log}
