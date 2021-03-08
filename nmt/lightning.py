# Copyright 2020 Skillfactory LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
from abc import ABC
from argparse import Namespace
from typing import List, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from nmt import data, model, tokenizer, layers


class LightningSequence2Sequence(pl.LightningModule, ABC):

    def __init__(self, hparams: Namespace):
        """
        Base module for sequence-to-sequence
        :param hparams: config of your project
        """
        super().__init__()

        self.hparams = hparams

        source_language_tokenizer, target_language_tokenizer = tokenizer.load_seq2seq_tokenizers(
            directory=self.hparams.directory)

        self.sequence2sequence_preparer = tokenizer.Sequence2SequencePreparer(
            source_language_tokenizer=source_language_tokenizer,
            target_language_tokenizer=target_language_tokenizer)

        self.model = model.Transformer(config=self.hparams)

        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.pad_index)
        self.criterion = layers.LabelSmoothingLoss(smoothing=self.hparams.smoothing,
                                                   use_kl=self.hparams.use_kl,
                                                   ignore_index=self.pad_index)

    def configure_optimizers(self):
        """
        Setup optimizer for training
        :return:
        """

        optimizer = torch.optim.AdamW(params=self.parameters(),
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        scheduler = layers.NoamScheduler(optimizer,
                                         model_dim=self.hparams.model_dim,
                                         warmup_steps=self.hparams.warmup_steps)

        return [optimizer], [scheduler]

    def get_dataloader(self, data_type: str, shuffle: bool = False) -> DataLoader:
        """
        Load data loader for training or validation
        :param data_type: setup training or validation
        :param shuffle: shuffle data at each epoch
        :return: DataLoader object
        """

        source_data = data.load_file(
            os.path.join(self.hparams.directory, f'{data_type}_en.txt'))
        target_data = data.load_file(
            os.path.join(self.hparams.directory, f'{data_type}_ru.txt'))

        dataset = data.Sequence2SequenceDataset(source_language_data=source_data,
                                                target_language_data=target_data)

        loader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.sequence2sequence_preparer.collate,
            shuffle=shuffle
        )

        return loader

    def train_dataloader(self):
        """
        Load train data loader
        :return: DataLoader object
        """
        loader = self.get_dataloader(data_type='train', shuffle=True)

        return loader

    def val_dataloader(self):
        """
        Load validation data loader
        :return: DataLoader object
        """
        loader = self.get_dataloader(data_type='valid', shuffle=False)

        return loader

    def forward(self,
                source_sequence: torch.Tensor,
                target_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence-to-sequence task
        :param source_sequence: batch of source texts indices
        :param target_sequence: batch of target texts indices
        :return: logits of your forward pass
        """

        logits = self.model(source_sequence, target_sequence)

        return logits

    # YOUR CODE STARTS
    def compute_loss(self, logits: torch.Tensor, target_criterion_ids: torch.Tensor) -> torch.Tensor:
        """
        How we compute loss for sequence-to-sequence task
        STEPS:
        1. Reshape logits and targets
        2. Compute loss
        :param logits: logits of your forward pass
        :param target_criterion_ids: target texts indices
        :return: loss scalar
        """
        prediction, target = logits.reshape(-1, logits.size(-1)), target_criterion_ids.contiguous().view(-1)

        loss = self.criterion(prediction, target)
        return loss
    # YOUR CODE ENDS

    def step(self, batch: (torch.Tensor, torch.Tensor, torch.Tensor)) -> torch.Tensor:
        """
        Step for training or validation
        :param batch: tuple of source texts ids, target texts ids and target ids for compute loss
        :return: loss scalar
        """

        source_ids, target_ids, target_criterion_ids = batch

        logits = self.forward(source_ids, target_ids)

        loss = self.compute_loss(logits, target_criterion_ids)

        return loss

    def training_step(self,
                      batch: (torch.Tensor, torch.Tensor, torch.Tensor),
                      batch_idx: int) -> Dict:
        """
        Do training step of your model
        :param batch: tuple of source texts ids, target texts ids and target ids for compute loss
        :param batch_idx: index of batch
        :return: Dict with info of that step
        """

        loss = self.step(batch=batch)

        self.log(name='train_loss', value=loss.item(),
                 prog_bar=False, on_step=True, on_epoch=False)
        self.log(name='train_perplexity', value=np.exp(loss.item()),
                 prog_bar=False, on_step=True, on_epoch=False)

        return {'loss': loss}

    def validation_step(self,
                        batch: (torch.Tensor, torch.Tensor, torch.Tensor),
                        batch_idx: int) -> Dict:
        """
        Do validation step of your model
        :param batch: tuple of source texts ids, target texts ids and target ids for compute loss
        :param batch_idx: index of batch
        :return: Dict with info of that step
        """

        loss = self.step(batch=batch)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> Dict:
        """
        What do we need to do at the end of validation iterations
        :param outputs: outputs of every validation step
        :return: Dict with info of all validation
        """
        epoch_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log(name='val_loss', value=epoch_loss.item(),
                 prog_bar=False, on_step=False, on_epoch=True)
        self.log(name='val_perplexity', value=np.exp(epoch_loss.item()),
                 prog_bar=False, on_step=False, on_epoch=True)

    def postprocessing(self, output_indices: List[List[int]]) -> List[List[int]]:

        output_indices = [sample[:sample.index(self.model.eos_index)]
                          if self.model.eos_index in sample
                          else sample
                          for sample in output_indices]

        return output_indices

    def generate(self, source_texts: List[str]) -> List[str]:
        """
        Translate from source texts (english) to target language (russian)
        :param source_texts: list of english texts
        :return: list of translated texts
        """

        tokenized = self.sequence2sequence_preparer.source_tokenize(source_texts)
        generated_indices = self.model.generate(source_text_ids=tokenized)
        generated_indices = self.postprocessing(output_indices=generated_indices)
        generated_texts = self.sequence2sequence_preparer.target_language_tokenizer.decode_batch(
            generated_indices)

        return generated_texts
