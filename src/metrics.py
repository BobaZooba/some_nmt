import os
import math
from src.data import load_file
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import pytorch_lightning as pl
from argparse import Namespace


# YOUR CODE STARTS
def calculate_bleu(lightning_model: pl.LightningModule, config: Namespace):

    valid_en = load_file(os.path.join(config.directory, 'valid_en.txt'))
    valid_ru = load_file(os.path.join(config.directory, 'valid_ru.txt'))

    predicted_texts = list()

    for i_batch in tqdm(range(math.ceil(len(valid_en) / config.batch_size)),
                        desc='Inference', disable=not config.verbose):

        batch = valid_en[i_batch * config.batch_size:(i_batch + 1) * config.batch_size]
        tokenized_batch = lightning_model.sequence2sequence_preparer.source_tokenize(batch)
        translated_batch = lightning_model.model.generate(tokenized_batch)
        predicted_texts_batch = lightning_model.sequence2sequence_preparer.target_language_tokenizer.decode_batch(
            translated_batch)
        predicted_texts.extend(predicted_texts_batch)

    tokenized_predicted = [word_tokenize(sample) for sample in predicted_texts]
    tokenized_target = [[word_tokenize(sample)] for sample in valid_ru]

    score = corpus_bleu(tokenized_target, tokenized_predicted)

    return score
# YOUR CODE ENDS
