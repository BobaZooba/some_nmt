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
import torch
from torch.utils.data import DataLoader
from nmt.data import Sequence2SequenceDataset, load_file
from nmt.tokenizer import Sequence2SequencePreparer, load_seq2seq_tokenizers
from tests.constants import TEST_DATA_DIRECTORY


def test_collate():
    en_tokenizer, ru_tokenizer = load_seq2seq_tokenizers(TEST_DATA_DIRECTORY)
    preparer = Sequence2SequencePreparer(en_tokenizer, ru_tokenizer)

    test_en = load_file(os.path.join(TEST_DATA_DIRECTORY, 'test_en.txt'))
    test_ru = load_file(os.path.join(TEST_DATA_DIRECTORY, 'test_ru.txt'))
    dataset = Sequence2SequenceDataset(test_en, test_ru)

    loader = DataLoader(dataset=dataset, batch_size=20, shuffle=False, collate_fn=preparer.collate)

    batch = next(iter(loader))

    assert len(batch) == 3
    assert type(batch) == tuple
    assert type(batch[0]) == torch.Tensor
    assert type(batch[1]) == torch.Tensor
    assert type(batch[2]) == torch.Tensor
    assert batch[0].shape == (20, 25)
    assert batch[1].shape == (20, 23)
    assert batch[2].shape == (20, 23)


def test_source_tokenize():
    en_tokenizer, ru_tokenizer = load_seq2seq_tokenizers(TEST_DATA_DIRECTORY)
    preparer = Sequence2SequencePreparer(en_tokenizer, ru_tokenizer)

    test_en = load_file(os.path.join(TEST_DATA_DIRECTORY, 'test_en.txt'))

    result = preparer.source_tokenize(test_en)

    assert type(result) == torch.Tensor
    assert result.shape == (20, 25)
