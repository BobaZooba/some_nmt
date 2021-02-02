from abc import ABC, abstractmethod
from argparse import Namespace
from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SpatialDropout(nn.Dropout2d):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x


class BaseSequence2Sequence(nn.Module, ABC):

    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.pad_index = self.config.pad_index
        self.bos_index = self.config.bos_index
        self.eos_index = self.config.eos_index

    @abstractmethod
    def init_weights(self):
        ...

    @abstractmethod
    def generate(self, seed: torch.Tensor) -> List[int]:
        ...

    def tensor_trimming(self, sequence: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        sequence_pad_mask = sequence != self.pad_index
        sequence_lengths = sequence_pad_mask.sum(dim=1)
        sequence_max_length = sequence_lengths.max()

        sequence = sequence[:, :sequence_max_length]
        sequence_pad_mask = sequence_pad_mask[:, :sequence_max_length]

        return sequence, sequence_pad_mask, sequence_lengths

    def sequence_length(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence_pad_mask = sequence != self.pad_index
        sequence_lengths = sequence_pad_mask.sum(dim=1)

        return sequence_lengths

    @abstractmethod
    def forward(self, source_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        ...


class Sequence2SequenceModel(BaseSequence2Sequence):

    def __init__(self, config: Namespace):
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

        self.init_weights()

    def init_weights(self):
        ...

    # YOUR CODE STARTS
    def forward(self, source_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:

        source_lengths = self.sequence_length(source_sequence)
        target_lengths = self.sequence_length(target_sequence)

        # embeddings
        source_emb = self.embedding_dropout(self.source_embedding_layer(source_sequence))
        target_emb = self.embedding_dropout(self.target_embedding_layer(target_sequence))

        # encoder
        packed_source_emb = pack_padded_sequence(source_emb,
                                                 source_lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)

        packed_encoded_sequence, encoder_mem = self.encoder_lstm(packed_source_emb)

        encoded_sequence, _ = pad_packed_sequence(packed_encoded_sequence, batch_first=True)

        # decoder
        packed_target_emb = pack_padded_sequence(target_emb,
                                                 target_lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)

        packed_decoded_sequence, _ = self.decoder_lstm(packed_target_emb, encoder_mem)

        decoded_sequence, _ = pad_packed_sequence(packed_decoded_sequence, batch_first=True)

        token_prediction = self.token_prediction_head(decoded_sequence)

        return token_prediction
    # YOUR CODE ENDS

    # YOUR CODE STARTS
    def generate(self, source_text_ids: torch.Tensor) -> List[int]:

        self.eval()

        output_indices = [list() for _ in range(source_text_ids.size(0))]

        with torch.no_grad():

            source_word_embeddings = self.source_embedding_layer(source_text_ids)

            source_lengths = self.sequence_length(source_text_ids)

            packed_source_emb = pack_padded_sequence(source_word_embeddings,
                                                     source_lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)

            _, memory = self.encoder_lstm(packed_source_emb)

            decoder_text_ids = torch.ones(source_text_ids.size(0), 1).long().to(source_word_embeddings.device)
            decoder_text_ids *= self.bos_index

            decoder_word_embeddings = self.target_embedding_layer(decoder_text_ids)

            for _ in range(self.config.max_length):

                decoder_hiddens, memory = self.decoder_lstm(decoder_word_embeddings, memory)

                token_logits = self.token_prediction_head(decoder_hiddens)

                token_predictions = torch.softmax(token_logits, -1).argmax(dim=-1)

                for n_sample in range(token_predictions.size(0)):
                    token_id = token_predictions[n_sample][0].item()
                    if token_id != self.eos_index:
                        output_indices[n_sample].append(token_id)

                decoder_word_embeddings = self.target_embedding_layer(token_predictions)

        return output_indices
    # YOUR CODE ENDS


class Sequence2SequenceWithAttentionModel(BaseSequence2Sequence):

    def __init__(self, config: Namespace):
        super().__init__(config=config)

        self.scaling = self.config.model_dim ** 0.5
        self.bidirectional_encoder = self.config.bidirectional_encoder

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
                                    batch_first=True,
                                    bidirectional=self.bidirectional_encoder)

        self.decoder_lstm = nn.LSTM(input_size=self.config.embedding_dim,
                                    hidden_size=self.config.model_dim,
                                    num_layers=self.config.decoder_num_layers,
                                    dropout=self.config.dropout,
                                    batch_first=True)

        self.query_projection = nn.Linear(in_features=self.config.model_dim, out_features=self.config.model_dim)
        self.key_projection = nn.Linear(in_features=self.config.model_dim * (int(self.bidirectional_encoder) + 1),
                                        out_features=self.config.model_dim)
        self.value_projection = nn.Linear(in_features=self.config.model_dim * (int(self.bidirectional_encoder) + 1),
                                          out_features=self.config.model_dim)

        self.attention_dropout = nn.Dropout(p=self.config.attention_dropout)

        self.attention_projection = nn.Linear(in_features=self.config.model_dim * 2,
                                              out_features=self.config.model_dim)

        self.token_prediction_head = torch.nn.Linear(in_features=self.config.model_dim,
                                                     out_features=self.config.vocab_size,
                                                     bias=False)

        if self.config.weight_tying and self.config.embedding_dim == self.config.model_dim:
            self.target_embedding_layer.weight = self.token_prediction_head.weight

        self.init_weights()

    def init_weights(self):
        ...

    def forward(self, source_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:

        source_sequence, source_pad_mask, source_lengths = self.tensor_trimming(source_sequence)
        target_lengths = self.sequence_length(target_sequence)

        # embeddings
        source_emb = self.embedding_dropout(self.source_embedding_layer(source_sequence))
        target_emb = self.embedding_dropout(self.target_embedding_layer(target_sequence))

        # encoder
        packed_source_emb = pack_padded_sequence(source_emb,
                                                 source_lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)

        packed_encoded_sequence, encoder_mem = self.encoder_lstm(packed_source_emb)

        if self.bidirectional_encoder:
            h_n = encoder_mem[0].view(self.encoder_lstm.num_layers, 2,
                                      source_sequence.size(0), encoder_mem[0].size(-1))

            c_n = encoder_mem[1].view(self.encoder_lstm.num_layers, 2,
                                      source_sequence.size(0), encoder_mem[1].size(-1))

            encoder_mem = (h_n[:, 0, :], c_n[:, 0, :])

        encoded_sequence, _ = pad_packed_sequence(packed_encoded_sequence, batch_first=True)

        # decoder
        packed_target_emb = pack_padded_sequence(target_emb,
                                                 target_lengths,
                                                 batch_first=True,
                                                 enforce_sorted=False)

        packed_decoded_sequence, _ = self.decoder_lstm(packed_target_emb, encoder_mem)

        decoded_sequence, _ = pad_packed_sequence(packed_decoded_sequence, batch_first=True)

        # attention
        query_emb = self.query_projection(decoded_sequence)
        key_emb = self.key_projection(encoded_sequence)
        value_emb = self.value_projection(encoded_sequence)

        attention_scores = torch.bmm(query_emb, key_emb.transpose(1, 2))
        attention_scores /= self.scaling
        attention_scores = attention_scores.masked_fill(
            ~source_pad_mask.unsqueeze(dim=1), -float('inf'))

        attention_distribution = torch.softmax(attention_scores, dim=-1)

        attention_vectors = torch.bmm(attention_distribution, value_emb)

        out_emb = torch.cat((decoded_sequence, attention_vectors), dim=2)

        out_emb = self.attention_projection(out_emb)

        token_prediction = self.token_prediction_head(out_emb)

        return token_prediction

    def generate(self, seed: torch.Tensor) -> torch.Tensor:
        ...

    def generate_single_response(self, source_sequence: torch.Tensor, max_seq_len=32) -> List[int]:

        self.eval()

        output_indices = list()

        with torch.no_grad():

            question_emb = self.source_embedding_layer(source_sequence)
            encoded_sequence, encoder_mem = self.encoder_lstm(question_emb)

            if self.bidirectional_encoder:
                h_n = encoder_mem[0].view(self.encoder_lstm.num_layers, 2,
                                          source_sequence.size(0), encoder_mem[0].size(-1))

                c_n = encoder_mem[1].view(self.encoder_lstm.num_layers, 2,
                                          source_sequence.size(0), encoder_mem[1].size(-1))

                mem = (h_n[:, 0, :].contiguous(), c_n[:, 0, :].contiguous())

            for timestemp in range(max_seq_len):

                decoded_sequence, decoder_mem = self.decoder_lstm(target_emb, mem)

                query_emb = self.query_projection(decoded_sequence)
                key_emb = self.key_projection(encoded_sequence)
                value_emb = self.value_projection(encoded_sequence)

                attention_scores = torch.bmm(query_emb, key_emb.transpose(1, 2))
                attention_scores /= self.scaling

                attention_distribution = torch.softmax(attention_scores, dim=-1)

                attention_vectors = torch.bmm(attention_distribution, value_emb)

                out_emb = torch.cat((decoded_sequence, attention_vectors), dim=2)

                out_emb = self.attention_projection(out_emb)

                token_prediction = self.token_prediction_head(out_emb)

                last_index = token_prediction[0, -1, :].argmax().item()

                if last_index == 3:
                    break

                target_sequence = torch.tensor([[last_index]]).to(source_sequence.device)

                target_emb = self.target_embedding_layer(target_sequence)

                output_indices.append(last_index)

        return output_indices
