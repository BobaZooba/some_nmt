from typing import List, Tuple

import torch
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
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
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    if add_bos_eos:
        special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]']
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )
    else:
        special_tokens = ['[PAD]', '[UNK]']

    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    tokenizer.train(train_files, trainer)

    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding()

    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    model_files = tokenizer.model.save(directory, prefix)
    tokenizer.model = BPE.from_file(*model_files, unk_token="[UNK]")

    tokenizer.save(save_path, pretty=True)

    return tokenizer


def train_seq2seq_tokenizers(directory: str,
                             en_source: bool = True,
                             max_length: int = 32,
                             vocab_size: int = 30_000):

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

    en_tokenizer = Tokenizer.from_file(os.path.join(directory, 'en_tokenizer.json'))
    ru_tokenizer = Tokenizer.from_file(os.path.join(directory, 'ru_tokenizer.json'))

    return en_tokenizer, ru_tokenizer


class Sequence2SequencePreparer:

    def __init__(self, source_language_tokenizer: Tokenizer, target_language_tokenizer: Tokenizer):

        self.source_language_tokenizer = source_language_tokenizer
        self.target_language_tokenizer = target_language_tokenizer

    def collate(self, batch: Tuple[List[str]]) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        source_texts: List[str] = list()
        target_texts: List[str] = list()

        for sample in batch:
            source_texts.append(sample[0])
            target_texts.append(sample[1])

        tokenized_source_texts = self.source_language_tokenizer.encode_batch(source_texts)
        tokenized_target_texts = self.target_language_tokenizer.encode_batch(source_texts)

        source_texts_ids: List[List[int]] = list()
        target_texts_ids: List[List[int]] = list()

        for sample_index in range(len(batch)):
            source_texts_ids.append(tokenized_source_texts[sample_index].ids)
            target_texts_ids.append(tokenized_target_texts[sample_index].ids)

        tensor_source_texts_ids: torch.Tensor = torch.tensor(source_texts_ids)
        tensor_target_texts_ids: torch.Tensor = torch.tensor(target_texts_ids)

        tensor_target_texts_ids_criterion: torch.Tensor = tensor_target_texts_ids[:, 1:]
        tensor_target_texts_ids = tensor_target_texts_ids[:, :-1]

        return tensor_source_texts_ids, tensor_target_texts_ids, tensor_target_texts_ids_criterion
