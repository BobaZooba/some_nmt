from src.model import Sequence2SequenceModel
import os
import torch
from torch.utils.data import DataLoader
from src.data import Sequence2SequenceDataset, load_file
from src.tokenizer import Sequence2SequencePreparer, load_seq2seq_tokenizers
from src.tests.constants import TEST_DATA_DIRECTORY, DEFAULT_CONFIG


def get_batch():
    en_tokenizer, ru_tokenizer = load_seq2seq_tokenizers(TEST_DATA_DIRECTORY)
    preparer = Sequence2SequencePreparer(en_tokenizer, ru_tokenizer)

    test_en = load_file(os.path.join(TEST_DATA_DIRECTORY, 'test_en.txt'))
    test_ru = load_file(os.path.join(TEST_DATA_DIRECTORY, 'test_ru.txt'))
    dataset = Sequence2SequenceDataset(test_en, test_ru)

    loader = DataLoader(dataset=dataset, batch_size=20, shuffle=False, collate_fn=preparer.collate)

    batch = next(iter(loader))

    return batch


def test_forward():

    tensor_source_texts_ids, tensor_target_texts_ids, tensor_target_texts_ids_criterion = get_batch()

    model = Sequence2SequenceModel(config=DEFAULT_CONFIG)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=DEFAULT_CONFIG.pad_index)

    model.eval()

    with torch.no_grad():
        logits = model(tensor_source_texts_ids, tensor_target_texts_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tensor_target_texts_ids_criterion.contiguous().view(-1))

    assert type(logits) == torch.Tensor
    assert logits.shape == (20, 23, DEFAULT_CONFIG.vocab_size)
    assert loss.shape == ()


def test_generate():

    tensor_source_texts_ids, _, _ = get_batch()

    model = Sequence2SequenceModel(config=DEFAULT_CONFIG)

    predicted_indices = model.generate(tensor_source_texts_ids)

    assert type(predicted_indices) == list
    assert len(predicted_indices[0]) == DEFAULT_CONFIG.max_length
