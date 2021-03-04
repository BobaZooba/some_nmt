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

from typing import List, Tuple

import torch
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def train_tokenizer(train_files: List[str],
                    save_path: str,
                    directory: str,
                    prefix: str,
                    add_bos_eos: bool = True,
                    max_length: int = 32,
                    vocab_size: int = 30_000) -> Tokenizer:
    """
    Train bpe tokenizer from files
    :param train_files: paths to files to train tokenizer
    :param save_path: where we need save tokenizer, .json
    :param directory: directory where we need save all information of tokenizer
    :param prefix: prefix to name of tokenizer
    :param add_bos_eos: do we need add bos and eos tokens
    :param max_length: maximum length of our texts, apply truncation if greater
    :param vocab_size: size of tokenizer vocabulary
    :return: trained tokenizer
    """
    tokenizer = Tokenizer(BPE(unk_token='[UNK]', end_of_word_suffix='</w>'))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = BPEDecoder()

    if add_bos_eos:
        special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]']
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )
    else:
        special_tokens = ['[PAD]', '[UNK]']

    trainer = BpeTrainer(vocab_size=vocab_size,
                         special_tokens=special_tokens,
                         end_of_word_suffix='</w>')

    tokenizer.train(train_files, trainer)

    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding()

    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    model_files = tokenizer.model.save(directory, prefix)
    tokenizer.model = BPE.from_file(*model_files, unk_token='[UNK]', end_of_word_suffix='</w>')

    tokenizer.save(save_path, pretty=True)

    return tokenizer


def train_seq2seq_tokenizers(directory: str,
                             en_source: bool = True,
                             max_length: int = 32,
                             vocab_size: int = 30_000):
    """
    Train two tokenizers: for english and russian
    :param directory: directory where we need save all information of tokenizer
    :param en_source: flag for english is our source text
    :param max_length: maximum length of our texts, apply truncation if greater
    :param vocab_size: size of tokenizer vocabulary
    :return: None
    """

    languages: List[str] = ['en', 'ru']

    if not en_source:
        languages = languages[::-1]

    for n, language in enumerate(languages):
        train_tokenizer(train_files=[os.path.join(directory, f'train_{language}.txt')],
                        save_path=os.path.join(directory, f'{language}_tokenizer.json'),
                        add_bos_eos=n > 0,
                        directory=directory,
                        prefix=language,
                        max_length=max_length,
                        vocab_size=vocab_size)


def load_seq2seq_tokenizers(directory: str) -> (Tokenizer, Tokenizer):
    """
    Loading of tokenizers from directory
    :param directory: directory with tokenizers
    :return: tokenizers
    """

    en_tokenizer = Tokenizer.from_file(os.path.join(directory, 'en_tokenizer.json'))
    ru_tokenizer = Tokenizer.from_file(os.path.join(directory, 'ru_tokenizer.json'))

    return en_tokenizer, ru_tokenizer


class Sequence2SequencePreparer:

    def __init__(self, source_language_tokenizer: Tokenizer, target_language_tokenizer: Tokenizer):
        """
        Module for handling samples and preparing it for model
        :param source_language_tokenizer: tokenizer for source text
        :param target_language_tokenizer: tokenizer for target text
        """

        self.source_language_tokenizer = source_language_tokenizer
        self.target_language_tokenizer = target_language_tokenizer

    # YOUR CODE STARTS
    def source_tokenize(self, batch: List[str]) -> torch.Tensor:
        """
        Tokenize source texts into indices
        STEPS:
        1. Tokenize using self.source_language_tokenizer
        2. Get .ids of each sample
        3. Turn it to tensor
        :param batch: list of strings
        :return: tensor with source texts indices, shape = (batch size, sequence length)
        """

        tokenized_source_texts = self.source_language_tokenizer.encode_batch(batch)

        source_texts_ids = [sample.ids for sample in tokenized_source_texts]

        tensor_source_texts_ids: torch.Tensor = torch.tensor(source_texts_ids)

        return tensor_source_texts_ids
    # YOUR CODE ENDS

    # YOUR CODE STARTS
    def collate(self, batch: Tuple[List[str]]) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Function that turn strings of source and target texts into tensor batches for model
        STEPS:
        1. Tokenize source and target texts using self.source_language_tokenizer and self.target_language_tokenizer
        2. Turn it into tensors
        3. Prepare tensor_target_texts_ids and tensor_target_texts_ids_criterion
        :param batch: tuple of lists source and target texts
        :return tensor_source_texts_ids: batch tensor of source texts indices
        :return tensor_target_texts_ids: batch tensor of target texts indices including <BOS> token index
        :return tensor_target_texts_ids_criterion: batch tensor of target texts indices including <EOS> token index
        """

        source_texts: List[str] = list()
        target_texts: List[str] = list()

        for sample in batch:
            source_texts.append(sample[0])
            target_texts.append(sample[1])

        tokenized_source_texts = self.source_language_tokenizer.encode_batch(source_texts)
        tokenized_target_texts = self.target_language_tokenizer.encode_batch(target_texts)

        source_texts_ids: List[List[int]] = list()
        target_texts_ids: List[List[int]] = list()

        for sample_index in range(len(batch)):
            if sum(tokenized_source_texts[sample_index].ids) > 0 and sum(tokenized_target_texts[sample_index].ids) > 0:
                source_texts_ids.append(tokenized_source_texts[sample_index].ids)
                target_texts_ids.append(tokenized_target_texts[sample_index].ids)

        tensor_source_texts_ids: torch.Tensor = torch.tensor(source_texts_ids)
        tensor_target_texts_ids: torch.Tensor = torch.tensor(target_texts_ids)

        tensor_target_texts_ids_criterion: torch.Tensor = tensor_target_texts_ids[:, 1:]
        tensor_target_texts_ids = tensor_target_texts_ids[:, :-1]

        return tensor_source_texts_ids, tensor_target_texts_ids, tensor_target_texts_ids_criterion
    # YOUR CODE ENDS
