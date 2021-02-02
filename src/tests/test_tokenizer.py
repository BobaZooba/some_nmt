import os
import torch
from torch.utils.data import DataLoader
from src.data import Sequence2SequenceDataset, load_file
from src.tokenizer import Sequence2SequencePreparer, load_seq2seq_tokenizers
from src.tests.constants import TEST_DATA_DIRECTORY


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
