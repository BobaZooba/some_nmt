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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nmt import activations


def embedding_masking(x: torch.Tensor,
                      pad_mask: torch.Tensor,
                      value: float = 0.) -> torch.Tensor:
    x = x.masked_fill((~pad_mask).unsqueeze(-1), value)
    return x


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


class GlobalMaskedPooling(nn.Module):
    POOLING_TYPES = ('mean', 'max')

    def __init__(self, pooling_type: str = 'mean', dim: int = 1, normalize: bool = False,
                 length_scaling: bool = False, scaling_square_root: bool = False):
        super().__init__()

        self.pooling_type = pooling_type
        self.dim = dim

        self.normalize = normalize
        self.length_scaling = length_scaling
        self.scaling_square_root = scaling_square_root

        if self.pooling_type == 'max':
            self.mask_value = -float('inf')
        else:
            self.mask_value = 0.

        if self.pooling_type not in self.POOLING_TYPES:
            raise ValueError(f'Available types: {", ".join(self.POOLING_TYPES)}')

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        lengths = pad_mask.sum(self.dim).float()

        x = embedding_masking(x=x, pad_mask=pad_mask, value=self.mask_value)

        if self.pooling_type == 'mean':
            scaling = x.size(self.dim) / lengths
        else:
            scaling = torch.ones(x.size(0))

        if self.length_scaling:
            lengths_factor = lengths
            if self.scaling_square_root:
                lengths_factor = lengths_factor ** 0.5
            scaling /= lengths_factor

        scaling = scaling.masked_fill(lengths == 0, 1.).unsqueeze(-1)

        if self.pooling_type == 'mean':
            x = x.mean(self.dim)
        else:
            x, _ = x.max(self.dim)

        x *= scaling

        if self.normalize:
            x = F.normalize(x)

        return x

    def extra_repr(self) -> str:

        description = [
            f'pooling_type="{self.pooling_type}"',
            f'normalize={self.normalize}',
            f'length_scaling={self.length_scaling}',
            f'scaling_square_root={self.scaling_square_root}',
        ]

        description = ', '.join(description)

        return description


class AttentionPooling(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 inner_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()

        inner_dim = model_dim if inner_dim is None else inner_dim

        self.key_projection = nn.Linear(in_features=model_dim, out_features=inner_dim)
        self.value_projection = nn.Linear(in_features=model_dim, out_features=inner_dim)

        self.pooling_projection = nn.Linear(in_features=inner_dim, out_features=num_heads, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.scaling = inner_dim ** 0.5

    def forward(self, x, pad_mask):

        key = self.key_projection(x)
        value = self.value_projection(x)

        key /= self.scaling

        attention_scores = self.pooling_projection(key).transpose(1, 2)

        attention_scores = attention_scores.masked_fill(~pad_mask.unsqueeze(1), -float('inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        x = torch.bmm(attention_scores, value)

        x = x.view(x.size(0), -1)

        return x


class FusionGate(nn.Module):

    def __init__(self, model_dim: int):
        super().__init__()

        self.raw_linear = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.hidden_linear = nn.Linear(in_features=model_dim, out_features=model_dim)

    def forward(self, raw: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.raw_linear(raw) + self.hidden_linear(hidden))

        x = gate * raw + (1 - gate) * hidden

        return x


class FactorizedEmbedding(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 token_embedding_dim: int,
                 padding_idx: int = 0,
                 zeroing_pad: bool = False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.zeroing_pad = zeroing_pad

        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=token_embedding_dim,
                                            padding_idx=padding_idx)

        self.projection = nn.Linear(in_features=token_embedding_dim, out_features=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        emb = self.embedding_layer(x)
        emb = self.projection(emb)

        if self.zeroing_pad:
            pad_mask = x != self.embedding_layer.padding_idx
            emb = embedding_masking(emb, pad_mask)

        return emb


class TransformerEmbedding(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 n_positions: int,
                 token_embedding_dim: Optional[int] = None,
                 n_segments: int = 1,
                 dropout: float = 0.1,
                 use_spatial_dropout: bool = False,
                 zeroing_pad: bool = True,
                 padding_idx: int = 0):
        super().__init__()

        self.zeroing_pad = zeroing_pad

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.token_embedding_dim = token_embedding_dim if token_embedding_dim is not None \
            else self.embedding_dim

        self.scaling = embedding_dim ** 0.5

        if self.token_embedding_dim != self.embedding_dim:
            self.token_embedding = FactorizedEmbedding(num_embeddings=vocab_size,
                                                       embedding_dim=self.embedding_dim,
                                                       token_embedding_dim=self.token_embedding_dim,
                                                       padding_idx=self.padding_idx,
                                                       zeroing_pad=False)
            self.is_factorized = True
        else:
            self.token_embedding = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=self.embedding_dim,
                                                padding_idx=self.padding_idx)
            self.is_factorized = False

        if n_segments > 2:
            self.segment_embedding = nn.Embedding(num_embeddings=n_segments,
                                                  embedding_dim=self.embedding_dim,
                                                  padding_idx=self.padding_idx)
        else:
            self.segment_embedding = None

        self.positional_embedding = nn.Embedding(num_embeddings=n_positions,
                                                 embedding_dim=self.embedding_dim,
                                                 padding_idx=self.padding_idx)

        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.dropout = SpatialDropout(p=dropout) if use_spatial_dropout else nn.Dropout(p=dropout)

    def forward(self,
                token_sequence: torch.Tensor,
                position_indices: Optional[torch.Tensor] = None,
                segment_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param token_sequence: [sequence_length, batch_size]
        :param position_indices: [sequence_length, batch_size]
        :param segment_indices: [sequence_length, batch_size]
        :return: [sequence_length, batch_size, model_dim]
        """

        emb = self.token_embedding(token_sequence) * self.scaling

        if position_indices is None:
            position_indices = torch.arange(1, token_sequence.size(1) + 1)
            position_indices = position_indices.unsqueeze(0).repeat(token_sequence.size(0), 1)
            position_indices = position_indices.to(token_sequence.device)

        position_emb = self.positional_embedding(position_indices)

        emb += position_emb

        if self.segment_embedding is not None:
            if segment_indices is None:
                segment_indices = torch.ones_like(token_sequence).to(token_sequence.device)

            segment_emb = self.segment_embedding(segment_indices)
            emb += segment_emb

        emb = self.dropout(self.layer_norm(emb))

        if self.zeroing_pad:
            pad_mask = token_sequence != self.padding_idx
            emb = embedding_masking(emb, pad_mask)

        return emb


class CausalCNN(nn.Module):

    def __init__(self,
                 model_dim: int,
                 kernel_size: int,
                 dropout: float = 0.3,
                 activation: Optional[str] = 'swish',
                 output_dim: Optional[int] = None):
        super().__init__()

        output_dim = output_dim if output_dim is not None else model_dim

        self.layer_norm = nn.LayerNorm(normalized_shape=model_dim)

        self.dropout = SpatialDropout(p=dropout)

        self.layer = nn.Conv1d(in_channels=model_dim, out_channels=output_dim, kernel_size=kernel_size)

        self.activation = activations.get_activation_function(activation=activation)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:

        residual = x

        x = self.layer_norm(x)

        x = self.dropout(x)

        x = F.pad(input=x.transpose(1, 2), pad=[self.layer.kernel_size[0] - 1, 0])

        x = self.layer(x).transpose(1, 2)

        x = self.activation(x)

        x = embedding_masking(x, pad_mask=pad_mask)

        x += residual

        return x


class ResidualLSTM(nn.Module):

    def __init__(self,
                 model_dim: int,
                 dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()

        self.bidirectional = bidirectional

        self.layer_norm = nn.LayerNorm(normalized_shape=model_dim)

        self.dropout = SpatialDropout(p=dropout)

        self.lstm = nn.LSTM(input_size=model_dim,
                            hidden_size=model_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.output_layer = nn.Sequential(nn.Dropout(p=dropout),
                                              nn.Linear(in_features=model_dim * 2,
                                                        out_features=model_dim))
        else:
            self.output_layer = nn.Identity()

    def forward(self,
                x: torch.Tensor,
                pad_mask: torch.Tensor,
                memory: Optional[Tuple[torch.Tensor,
                                       torch.Tensor]] = None) -> Tuple[torch.Tensor,
                                                                       torch.Tensor]:

        residual = x

        x = self.layer_norm(x)
        x = embedding_masking(x=x, pad_mask=pad_mask)

        x = self.dropout(x)

        x_packed = pack_padded_sequence(x,
                                        pad_mask.sum(1),
                                        batch_first=True,
                                        enforce_sorted=False)

        x_packed, memory = self.lstm(x_packed, memory)

        x, _ = pad_packed_sequence(x_packed,
                                   batch_first=True,
                                   total_length=x.size(1))

        x = self.output_layer(x)

        x = embedding_masking(x=x, pad_mask=pad_mask)

        x = x + residual

        return x, memory


class MultiHeadSelfAttention(nn.Module):
    """
    Need re-implement this layer for custom attention mask with shape: [batch_size, sequence_len, sequence_len]
    """

    def __init__(self, model_dim: int, num_heads: int, head_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()

        self.num_heads = num_heads

        if head_dim is None:
            self.head_dim = model_dim // num_heads
            self.layer_dim = model_dim
        else:
            self.head_dim = head_dim
            self.layer_dim = self.head_dim * self.num_heads

        self.scaling = self.head_dim ** 0.5

        self.in_projection = nn.Linear(model_dim, 3 * self.layer_dim)
        self.out_projection = nn.Linear(self.layer_dim, model_dim)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self,
                    embed: torch.Tensor,
                    batch_size: int,
                    sequence_len: int) -> torch.Tensor:
        """
        From [batch_size * self.num_heads, sequence_len, sequence_len]
        To [batch_size, self.num_heads, sequence_len, sequence_len]
        """
        return embed.view(batch_size, self.num_heads, sequence_len, sequence_len)

    def join_heads(self,
                   embed: torch.Tensor,
                   batch_size: int,
                   sequence_len: int) -> torch.Tensor:
        """
        From [batch_size, self.num_heads, sequence_len, sequence_len]
        To [batch_size * self.num_heads, sequence_len, sequence_len]
        """
        return embed.view(batch_size * self.num_heads, sequence_len, sequence_len)

    def forward(self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param padding_mask: [batch_size, sequence_len]
        :param attention_mask: [batch_size, sequence_len, sequence_len]
        :param need_weights: bool
        :return: [batch_size, sequence_length, model_dim]
        """

        x = x.transpose(0, 1)

        sequence_len, batch_size, model_dim = x.size()

        query, key, value = self.in_projection(x).chunk(3, dim=-1)

        if self.num_heads > 1:
            # [batch_size * self.num_heads, sequence_len, self.head_dim]
            query = query.contiguous().view(sequence_len, batch_size * self.num_heads, self.head_dim)
            key = key.contiguous().view(sequence_len, batch_size * self.num_heads, self.head_dim)
            value = value.contiguous().view(sequence_len, batch_size * self.num_heads, self.head_dim)

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        query /= self.scaling

        # [batch_size * self.num_heads, sequence_len, sequence_len]
        attention_scores = torch.bmm(query, key.transpose(1, 2))

        if self.num_heads > 1:
            # [batch_size, self.num_heads, sequence_len, sequence_len]
            attention_scores = self.split_heads(attention_scores, batch_size, sequence_len)

        # fp16 compatibility
        parameters_type = next(self.parameters()).dtype

        if attention_mask is not None:
            if self.num_heads > 1:
                # [batch_size, sequence_len, sequence_len] -> [batch_size, 1, sequence_len, sequence_len]
                attention_mask = attention_mask.unsqueeze(1)

            attention_mask = attention_mask.to(dtype=parameters_type)
            attention_scores += attention_mask

        if padding_mask is not None:
            # [batch_size, sequence_len] -> [batch_size, 1, sequence_len]
            padding_mask = ~padding_mask.unsqueeze(1)

            if self.num_heads > 1:
                # [batch_size, 1, sequence_len] -> [batch_size, 1, 1, sequence_len]
                padding_mask = padding_mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(
                padding_mask,
                -float('inf'),
            )

        if self.num_heads > 1:
            # [batch_size * self.num_heads, sequence_len, sequence_len]
            attention_scores = self.join_heads(attention_scores, batch_size, sequence_len)

        if attention_scores.dtype == torch.float16:
            tensor_type = torch.float32
        else:
            tensor_type = attention_scores.dtype

        # [batch_size * self.num_heads, sequence_len, sequence_len]
        attention_scores = F.softmax(attention_scores.float(), dim=-1, dtype=tensor_type)

        attention_scores = self.dropout(attention_scores)

        # attention_scores = [batch_size * self.num_heads, sequence_len, sequence_len]
        # value = [batch_size * self.num_heads, sequence_len, self.head_dim]
        # [batch_size * self.num_heads, sequence_len, self.head_dim]
        attention_output = torch.bmm(attention_scores, value)

        # [sequence_len, batch_size, model_dim]
        attention_output = attention_output.transpose(0, 1).contiguous().view(sequence_len,
                                                                              batch_size,
                                                                              self.layer_dim)

        attention_output = self.out_projection(attention_output)

        # for visualize attention scores
        if need_weights:
            if self.num_heads > 1:
                # [batch_size, self.num_heads, sequence_len, sequence_len]
                attention_scores = self.split_heads(attention_scores, batch_size, sequence_len)
        else:
            attention_scores = None

        attention_output = attention_output.transpose(0, 1)

        return attention_output, attention_scores


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 increased_dim: int,
                 dropout: float = 0.1,
                 activation: str = 'swish',
                 output_dim: Optional[int] = None):
        super().__init__()

        output_dim = output_dim if output_dim is not None else model_dim

        self.increase = nn.Linear(in_features=model_dim, out_features=increased_dim)
        self.activation = activations.get_activation_function(activation=activation)
        self.dropout = nn.Dropout(dropout)
        self.decrease = nn.Linear(in_features=increased_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [*, model_dim]
        :return: [*, model_dim]
        """

        x = self.increase(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.decrease(x)

        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 activation='swish',
                 use_fusion_gate: bool = False):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(model_dim=model_dim,
                                                     num_heads=num_heads,
                                                     head_dim=head_dim,
                                                     dropout=dropout)

        self.attention_dropout = nn.Dropout(dropout)

        self.fusion_gate = FusionGate(model_dim=model_dim) if use_fusion_gate else None

        self.attention_norm = nn.LayerNorm(model_dim)

        self.position_wise_feed_forward = PositionWiseFeedForwardLayer(model_dim=model_dim,
                                                                       increased_dim=feed_forward_dim,
                                                                       dropout=dropout,
                                                                       activation=activation)

        self.feed_forward_dropout = nn.Dropout(dropout)

        self.feed_forward_norm = nn.LayerNorm(model_dim)

    def forward(self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: [batch_size, sequence_length, model_dim]
        :param padding_mask: [batch_size, sequence_len]
        :param attention_mask: [batch_size, sequence_len, sequence_len]
        :return: [batch_size, sequence_length, model_dim]
        """

        hidden, _ = self.self_attention(x=x,
                                        padding_mask=padding_mask,
                                        attention_mask=attention_mask)

        hidden = self.attention_dropout(hidden)

        if self.fusion_gate is not None:
            x = self.fusion_gate(x, hidden)
        else:
            x = x + hidden

        x = self.attention_norm(x)

        hidden = self.position_wise_feed_forward(x)

        x = x + self.feed_forward_dropout(hidden)

        x = self.feed_forward_norm(x)

        return x


class LSTMTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 n_layers: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 head_dim: Optional[int] = None,
                 rnn_dropout: float = 0.3,
                 transformer_dropout: float = 0.1,
                 activation='swish',
                 use_fusion_gate: bool = False):
        super().__init__()

        self.lstm = ResidualLSTM(model_dim=model_dim, dropout=rnn_dropout, bidirectional=True)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim=model_dim,
                                    num_heads=num_heads,
                                    feed_forward_dim=feed_forward_dim,
                                    head_dim=head_dim,
                                    dropout=transformer_dropout,
                                    activation=activation,
                                    use_fusion_gate=use_fusion_gate)
            for _ in range(n_layers)])

    def forward(self,
                x: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x, memory = self.input_lstm(x, pad_mask)

        for layer in self.encoder_layers:
            x = layer(x, pad_mask, attention_mask)

        x = embedding_masking(x, pad_mask)

        return x, memory


class TransformerDecoderLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 head_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 activation='swish',
                 use_fusion_gate: bool = False):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(model_dim=model_dim,
                                                     num_heads=num_heads,
                                                     head_dim=head_dim,
                                                     dropout=dropout)

        self.self_attention_dropout = nn.Dropout(dropout)

        self.self_attention_fusion_gate = FusionGate(model_dim=model_dim) if use_fusion_gate else None

        self.self_attention_norm = nn.LayerNorm(model_dim)

        self.attention = nn.MultiheadAttention(embed_dim=model_dim,
                                               num_heads=num_heads,
                                               dropout=dropout)

        self.attention_dropout = nn.Dropout(dropout)

        self.attention_fusion_gate = FusionGate(model_dim=model_dim) if use_fusion_gate else None

        self.attention_norm = nn.LayerNorm(model_dim)

        self.position_wise_feed_forward = PositionWiseFeedForwardLayer(model_dim=model_dim,
                                                                       increased_dim=feed_forward_dim,
                                                                       dropout=dropout,
                                                                       activation=activation)

        self.feed_forward_dropout = nn.Dropout(dropout)

        self.feed_forward_norm = nn.LayerNorm(model_dim)

    def forward(self,
                source_sequence: torch.Tensor,
                target_sequence: torch.Tensor,
                source_padding_mask: Optional[torch.Tensor] = None,
                target_padding_mask: Optional[torch.Tensor] = None,
                target_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param source_sequence:
        :param target_sequence:
        :param source_padding_mask:
        :param target_padding_mask:
        :param source_attention_mask:
        :return:
        """
        # """
        # :param x: [batch_size, sequence_length, model_dim]
        # :param padding_mask: [batch_size, sequence_len]
        # :param attention_mask: [batch_size, sequence_len, sequence_len]
        # :return: [batch_size, sequence_length, model_dim]
        # """

        hidden, _ = self.self_attention(x=target_sequence,
                                        padding_mask=target_padding_mask,
                                        attention_mask=target_attention_mask)

        hidden = self.self_attention_dropout(hidden)

        if self.self_attention_fusion_gate is not None:
            target_sequence = self.self_attention_fusion_gate(target_sequence, hidden)
        else:
            target_sequence = target_sequence + hidden

        target_sequence = self.self_attention_norm(target_sequence)

        target_sequence = target_sequence.transpose(0, 1)
        source_sequence = source_sequence.transpose(0, 1)

        hidden, _ = self.attention(target_sequence,
                                   source_sequence,
                                   source_sequence,
                                   key_padding_mask=source_padding_mask)

        hidden = hidden.transpose(0, 1)

        hidden = self.attention_dropout(hidden)

        if self.self_attention_fusion_gate is not None:
            target_sequence = self.attention_fusion_gate(target_sequence, hidden)
        else:
            target_sequence = target_sequence + hidden

        target_sequence = self.attention_norm(target_sequence)

        hidden = self.position_wise_feed_forward(target_sequence)

        target_sequence = target_sequence + self.feed_forward_dropout(hidden)

        target_sequence = self.feed_forward_norm(target_sequence)

        return target_sequence


class LSTMTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 model_dim: int,
                 n_layers: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 head_dim: Optional[int] = None,
                 rnn_dropout: float = 0.3,
                 transformer_dropout: float = 0.1,
                 activation='swish',
                 use_fusion_gate: bool = False):
        super().__init__()

        self.lstm = ResidualLSTM(model_dim=model_dim, dropout=rnn_dropout, bidirectional=False)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim=model_dim,
                                    num_heads=num_heads,
                                    feed_forward_dim=feed_forward_dim,
                                    head_dim=head_dim,
                                    dropout=transformer_dropout,
                                    activation=activation,
                                    use_fusion_gate=use_fusion_gate)
            for _ in range(n_layers)])

    def forward(self,
                source_sequence: torch.Tensor,
                target_sequence: torch.Tensor,
                encoder_memory: Optional[torch.Tensor] = None,
                source_padding_mask: Optional[torch.Tensor] = None,
                target_padding_mask: Optional[torch.Tensor] = None,
                target_attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        target_sequence, memory = self.input_lstm(target_sequence, target_padding_mask, encoder_memory)

        for layer in self.decoder_layers:
            target_sequence = layer(source_sequence,
                                    target_sequence,
                                    source_padding_mask,
                                    target_padding_mask,
                                    target_attention_mask)

        target_sequence = embedding_masking(target_sequence, target_padding_mask)

        return target_sequence, memory
