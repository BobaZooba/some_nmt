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

from nmt import layers


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

    def tensor_trimming(self, sequence: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Trim off excess length for more effective inference of your model
        :param sequence: batch of text indices, shape = (batch size, sequence length)
        :return sequence: trimmed sequence, shape = (batch size, sequence length)
        :return sequence_pad_mask: bool tensor with 1 for text and 0 for pad, shape = (batch size, sequence length)
        :return sequence_lengths: tensor with lengths of every sample with no pad, shape = (batch size)
        """
        sequence_pad_mask = sequence != self.pad_index
        sequence_lengths = sequence_pad_mask.sum(dim=1).to(sequence.device)
        sequence_max_length = sequence_lengths.max()

        sequence = sequence[:, :sequence_max_length]
        sequence_pad_mask = sequence_pad_mask[:, :sequence_max_length].to(sequence.device)

        return sequence, sequence_pad_mask, sequence_lengths

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


class Sequence2SequenceWithAttentionModel(BaseSequence2Sequence):

    def __init__(self, config: Namespace):
        """
        Module for sequence-to-sequence task
        :param config: hyper parameters of your experiment
        """
        super().__init__(config=config)

        self.bidirectional_encoder = self.config.bidirectional_encoder

        self.source_embedding_layer = nn.Embedding(num_embeddings=self.config.vocab_size,
                                                   embedding_dim=self.config.embedding_dim,
                                                   padding_idx=self.pad_index)

        self.target_embedding_layer = nn.Embedding(num_embeddings=self.config.vocab_size,
                                                   embedding_dim=self.config.embedding_dim,
                                                   padding_idx=self.pad_index)

        self.embedding_dropout = layers.SpatialDropout(p=self.config.dropout)

        self.encoder_lstm = nn.LSTM(input_size=self.config.embedding_dim,
                                    hidden_size=self.config.model_dim,
                                    num_layers=self.config.encoder_num_layers,
                                    dropout=self.config.dropout,
                                    batch_first=True,
                                    bidirectional=self.bidirectional_encoder)

        self.decoder_lstm = nn.LSTM(input_size=self.config.embedding_dim,
                                    hidden_size=self.config.model_dim,
                                    num_layers=self.config.decoder_num_layers,
                                    dropout=self.config.dropout,
                                    batch_first=True)

        self.attention_dropout = nn.Dropout(p=self.config.attention_dropout)

        self.query_projection = nn.Linear(in_features=self.config.model_dim, out_features=self.config.model_dim)
        self.key_projection = nn.Linear(in_features=self.config.model_dim * (int(self.bidirectional_encoder) + 1),
                                        out_features=self.config.model_dim)
        self.value_projection = nn.Linear(in_features=self.config.model_dim * (int(self.bidirectional_encoder) + 1),
                                          out_features=self.config.model_dim)

        self.attention_projection = nn.Linear(in_features=self.config.model_dim * 2,
                                              out_features=self.config.model_dim)

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
            5.1* If you using bidirectional encoder reshape memory and get forward pass to put it into decoder
        6. Unpack lstm results with pad_packed_sequence
        7. Pack target text using pack_padded_sequence
        8. Apply decoder lstm for packed target texts and using encoder lstm memory
        9. Unpack lstm results with pad_packed_sequence
        10. Compute the attention
            10.1. Compute attention scores between decoder and encoder lstm hiddens
            10.2. Compute attention distribution using softmax
            10.3. Compute attention vectors using attention distribution and decoder lstm hiddens
            10.4. Aggregate attention vectors with decoder lstm hiddens
            10.5.* You also can use some of linear projections for decoder and encoder hiddens
            10.6.* You can use linear projection for attention vectors
        11. Predict the words of target texts
        :param source_sequence: batch of source texts indices
        :param target_sequence: batch of target texts indices
        :return: logits of your forward pass
        """

        source_sequence, source_pad_mask, source_lengths = self.tensor_trimming(source_sequence)
        target_lengths = self.sequence_length(target_sequence)

        # embeddings
        source_emb = self.embedding_dropout(self.source_embedding_layer(source_sequence))
        target_emb = self.embedding_dropout(self.target_embedding_layer(target_sequence))

        # encoder
        packed_source_emb = pack_padded_sequence(source_emb,
                                                 source_lengths.cpu(),
                                                 batch_first=True,
                                                 enforce_sorted=False)

        packed_encoded_sequence, encoder_memory = self.encoder_lstm(packed_source_emb)

        if self.bidirectional_encoder:
            h_n = encoder_memory[0].view(self.encoder_lstm.num_layers, 2,
                                         source_sequence.size(0), encoder_memory[0].size(-1))

            c_n = encoder_memory[1].view(self.encoder_lstm.num_layers, 2,
                                         source_sequence.size(0), encoder_memory[1].size(-1))

            encoder_memory = (h_n[:, 0, :], c_n[:, 0, :])

        encoded_sequence, _ = pad_packed_sequence(packed_encoded_sequence, batch_first=True)

        # decoder
        packed_target_emb = pack_padded_sequence(target_emb,
                                                 target_lengths.cpu(),
                                                 batch_first=True,
                                                 enforce_sorted=False)

        packed_decoded_sequence, _ = self.decoder_lstm(packed_target_emb, encoder_memory)

        decoded_sequence, _ = pad_packed_sequence(packed_decoded_sequence, batch_first=True)

        # attention
        query_emb = self.query_projection(decoded_sequence)
        key_emb = self.key_projection(encoded_sequence)
        value_emb = self.value_projection(encoded_sequence)

        attention_scores = torch.bmm(query_emb, key_emb.transpose(1, 2))

        attention_scores = attention_scores.masked_fill(
            ~source_pad_mask.unsqueeze(1),
            -float('inf'),
        )

        attention_distribution = torch.softmax(attention_scores, dim=-1)

        attention_distribution = self.attention_dropout(attention_distribution)

        attention_vectors = torch.bmm(attention_distribution, value_emb)

        out_emb = torch.cat((decoded_sequence, attention_vectors), dim=2)

        out_emb = self.attention_projection(out_emb)

        token_prediction = self.token_prediction_head(out_emb)

        return token_prediction
    # YOUR CODE ENDS

    # YOUR CODE STARTS
    def generate(self, source_text_ids: torch.Tensor) -> List[List[int]]:
        """
        Function that generate translation
        STEPS:
        1. Turn model to evaluation mode
        2. Apply encoder things (use code from forward step)
        3. Use loop to predict every token of translation, dont forget about <BOS> token
            3.1. You need update your memory at each step
            3.2. Use updated memory to inference for new step
            3.3. Apply attention from current decoder step to every encoder hiddens
        4. Save results to list
        :param source_text_ids: batch of source texts indices
        :return: batch of predicted translation, indices
        """

        self.eval()

        output_indices = [list() for _ in range(source_text_ids.size(0))]

        with torch.no_grad():

            source_text_ids, source_pad_mask, source_lengths = self.tensor_trimming(source_text_ids)

            source_word_embeddings = self.source_embedding_layer(source_text_ids)

            packed_source_emb = pack_padded_sequence(source_word_embeddings,
                                                     source_lengths.cpu(),
                                                     batch_first=True,
                                                     enforce_sorted=False)

            packed_encoded_sequence, memory = self.encoder_lstm(packed_source_emb)

            if self.bidirectional_encoder:
                h_n = memory[0].view(self.encoder_lstm.num_layers, 2,
                                     source_text_ids.size(0), memory[0].size(-1))

                c_n = memory[1].view(self.encoder_lstm.num_layers, 2,
                                     source_text_ids.size(0), memory[1].size(-1))

                memory = (h_n[:, 0, :], c_n[:, 0, :])

            encoded_sequence, _ = pad_packed_sequence(packed_encoded_sequence, batch_first=True)

            decoder_text_ids = torch.ones(source_text_ids.size(0), 1).long().to(source_word_embeddings.device)
            decoder_text_ids *= self.bos_index

            decoder_word_embeddings = self.target_embedding_layer(decoder_text_ids)

            for _ in range(self.config.max_length):

                decoded_sequence, memory = self.decoder_lstm(decoder_word_embeddings, memory)

                query_emb = self.query_projection(decoded_sequence)
                key_emb = self.key_projection(encoded_sequence)
                value_emb = self.value_projection(encoded_sequence)

                attention_scores = torch.bmm(query_emb, key_emb.transpose(1, 2))

                attention_scores = attention_scores.masked_fill(
                    ~source_pad_mask.unsqueeze(1),
                    -float('inf'),
                )

                attention_distribution = torch.softmax(attention_scores, dim=-1)
                attention_vectors = torch.bmm(attention_distribution, value_emb)

                out_emb = torch.cat((decoded_sequence, attention_vectors), dim=2)

                out_emb = self.attention_projection(out_emb)

                token_logits = self.token_prediction_head(out_emb)
                token_predictions = torch.softmax(token_logits, -1).argmax(dim=-1)

                for n_sample in range(token_predictions.size(0)):
                    token_id = token_predictions[n_sample][0].item()
                    output_indices[n_sample].append(token_id)

                decoder_word_embeddings = self.target_embedding_layer(token_predictions)

        output_indices = [sample[:sample.index(self.eos_index)]
                          if self.eos_index in sample
                          else sample
                          for sample in output_indices]

        return output_indices
    # YOUR CODE ENDS


class Transformer(BaseSequence2Sequence):

    def __init__(self, config: Namespace):
        super().__init__(config=config)

        self.source_embedding_layer = layers.TransformerEmbedding(embedding_dim=self.config.model_dim,
                                                                  vocab_size=self.config.vocab_size,
                                                                  n_positions=self.config.max_length + 1,
                                                                  token_embedding_dim=self.config.embedding_dim,
                                                                  dropout=self.config.embedding_dropout,
                                                                  use_spatial_dropout=False,
                                                                  padding_idx=self.pad_index,
                                                                  zeroing_pad=True)

        self.source_embedding_input_cnn = layers.CausalCNN(model_dim=self.config.model_dim,
                                                           kernel_size=self.config.input_cnn_kernel_size,
                                                           dropout=self.config.embedding_dropout,
                                                           activation=self.config.activation)

        self.target_embedding_layer = layers.TransformerEmbedding(embedding_dim=self.config.model_dim,
                                                                  vocab_size=self.config.vocab_size,
                                                                  n_positions=self.config.max_length + 1,
                                                                  token_embedding_dim=self.config.embedding_dim,
                                                                  dropout=self.config.embedding_dropout,
                                                                  use_spatial_dropout=False,
                                                                  padding_idx=self.pad_index,
                                                                  zeroing_pad=True)

        self.target_embedding_input_cnn = layers.CausalCNN(model_dim=self.config.model_dim,
                                                           kernel_size=self.config.input_cnn_kernel_size,
                                                           dropout=self.config.embedding_dropout,
                                                           activation=self.config.activation)

        self.encoder_layers = nn.ModuleList([
            layers.TransformerEncoderLayer(model_dim=self.config.model_dim,
                                           num_heads=self.config.num_heads,
                                           feed_forward_dim=self.config.feed_forward_dim,
                                           head_dim=self.config.head_dim,
                                           dropout=self.config.transformer_dropout,
                                           activation=self.config.activation,
                                           use_fusion_gate=self.config.use_fusion_gate)
            for _ in range(self.config.encoder_num_layers)])

        self.decoder_layers = nn.ModuleList([
            layers.TransformerDecoderLayer(model_dim=self.config.model_dim,
                                           num_heads=self.config.num_heads,
                                           feed_forward_dim=self.config.feed_forward_dim,
                                           head_dim=self.config.head_dim,
                                           dropout=self.config.transformer_dropout,
                                           activation=self.config.activation,
                                           use_fusion_gate=self.config.use_fusion_gate)
            for _ in range(self.config.decoder_num_layers)])

        self.token_prediction_head = torch.nn.Linear(in_features=self.config.model_dim,
                                                     out_features=self.config.vocab_size,
                                                     bias=False)

        if self.config.weight_tying and not self.target_embedding_layer.is_factorized:
            self.token_prediction_head.weight = self.target_embedding_layer.token_embedding.weight

    def forward(self, source_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:

        source_sequence, source_pad_mask, _ = self.tensor_trimming(source_sequence)
        target_sequence, target_pad_mask, _ = self.tensor_trimming(target_sequence)

        # embeddings
        source_sequence = self.source_embedding_layer(source_sequence)
        source_sequence = self.source_embedding_input_cnn(source_sequence, source_pad_mask)

        target_sequence = self.target_embedding_layer(target_sequence)
        target_sequence = self.target_embedding_input_cnn(target_sequence, target_pad_mask)

        # encoder
        for layer in self.encoder_layers:
            source_sequence = layer(source_sequence, source_pad_mask)

        # decoder
        for layer in self.decoder_layers:
            target_sequence = layer(source_sequence,
                                    target_sequence,
                                    source_pad_mask,
                                    target_pad_mask)

        # prediction
        token_prediction = self.token_prediction_head(target_sequence)

        return token_prediction

    def generate(self, source_text_ids: torch.Tensor) -> List[List[int]]:

        source_sequence, source_pad_mask, _ = self.tensor_trimming(source_text_ids)
        # target_sequence, target_pad_mask, _ = self.tensor_trimming(target_sequence)

        # embeddings
        source_sequence = self.source_embedding_layer(source_sequence)
        source_sequence = self.source_embedding_input_cnn(source_sequence, source_pad_mask)

        # target_sequence = self.target_embedding_layer(target_sequence)
        # target_sequence = self.target_embedding_input_cnn(target_sequence, target_pad_mask)

        # encoder
        for layer in self.encoder_layers:
            source_sequence = layer(source_sequence, source_pad_mask)

        return []
