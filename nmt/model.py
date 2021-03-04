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

from abc import ABC, abstractmethod
from argparse import Namespace
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpatialDropout(nn.Dropout2d):

    def __init__(self, p=0.5):
        """
        Apply special dropout that work cool for rnn models
        :param p: probability of dropout
        """
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the dropout
        :param x: tensor with word vectors, shape = (batch size, sequence length, vector dim)
        :return: tensor after dropout
        """
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x


class BaseSequence2Sequence(nn.Module, ABC):

    def __init__(self, config: Namespace):
        """
        Base module for sequence-to-sequence task
        :param config: hyper parameters of your experiment
        """
        super().__init__()
        self.config = config
        self.pad_index = self.config.pad_index
        self.bos_index = self.config.bos_index
        self.eos_index = self.config.eos_index

    @abstractmethod
    def generate(self, source_text_ids: torch.Tensor) -> List[List[int]]:
        """
        Function that generate translation
        :param source_text_ids: batch of source texts indices
        :return: batch of predicted translation, indices
        """
        ...

    def sequence_length(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence length with no pad
        :param sequence: batch of text indices, shape = (batch size, sequence length)
        :return: tensor with lengths of every sample with no pad, shape = (batch size)
        """
        sequence_pad_mask = sequence != self.pad_index
        sequence_lengths = sequence_pad_mask.sum(dim=1).to(sequence.device)

        return sequence_lengths

    @abstractmethod
    def forward(self, source_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence-to-sequence task
        :param source_sequence: batch of source texts indices
        :param target_sequence: batch of target texts indices
        :return: logits of your forward pass
        """
        ...


class Sequence2SequenceModel(BaseSequence2Sequence):

    def __init__(self, config: Namespace):
        """
        Module for sequence-to-sequence task
        :param config: hyper parameters of your experiment
        """
        super().__init__(config=config)

        self.source_embedding_layer = nn.Embedding(num_embeddings=self.config.vocab_size,
                                                   embedding_dim=self.config.embedding_dim,
                                                   padding_idx=self.pad_index)

        self.target_embedding_layer = nn.Embedding(num_embeddings=self.config.vocab_size,
                                                   embedding_dim=self.config.embedding_dim,
                                                   padding_idx=self.pad_index)

        self.embedding_dropout = SpatialDropout(p=self.config.dropout)

        self.encoder_lstm = nn.LSTM(input_size=self.config.embedding_dim,
                                    hidden_size=self.config.model_dim,
                                    num_layers=self.config.encoder_num_layers,
                                    dropout=self.config.dropout,
                                    batch_first=True)

        self.decoder_lstm = nn.LSTM(input_size=self.config.embedding_dim,
                                    hidden_size=self.config.model_dim,
                                    num_layers=self.config.decoder_num_layers,
                                    dropout=self.config.dropout,
                                    batch_first=True)

        self.token_prediction_head = torch.nn.Linear(in_features=self.config.model_dim,
                                                     out_features=self.config.vocab_size,
                                                     bias=False)

        if self.config.weight_tying and self.config.embedding_dim == self.config.model_dim:
            self.target_embedding_layer.weight = self.token_prediction_head.weight

    # YOUR CODE STARTS
    def forward(self, source_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence-to-sequence task
        STEPS:
        1. Compute lengths
        2. Turn to word embeddings source and target batch
        3. Apply dropout
        4. Pack source text using pack_padded_sequence
        5. Apply encoder lstm for packed source texts
        6. Unpack lstm results with pad_packed_sequence
        7. Pack target text using pack_padded_sequence
        8. Apply decoder lstm for packed target texts and using encoder lstm memory
        9. Unpack lstm results with pad_packed_sequence
        10. Predict the words of target texts
        :param source_sequence: batch of source texts indices
        :param target_sequence: batch of target texts indices
        :return: logits of your forward pass
        """

        ...

    # YOUR CODE ENDS

    # YOUR CODE STARTS
    def generate(self, source_text_ids: torch.Tensor) -> List[List[int]]:
        """
        Function that generate translation. This function should work with batches
        STEPS:
        1. Turn model to evaluation mode
        2. Apply encoder things (use code from forward step)
        3. Use loop to predict every token of translation, dont forget about <BOS> token
            3.1. You need update your memory at each step
            3.2. Use updated memory to inference for new step
        4. Save results to list
            4.1. Don't forget about EOS index. You can stop generation when you see that token or delete them latter
            4.2. Don't forget that you don't need any generated tokens after the EOS token
        :param source_text_ids: batch of source texts indices
        :return: batch of predicted translation, indices
        """

        ...

    # YOUR CODE ENDS
