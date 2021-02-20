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
import math
from nmt.data import load_file
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import pytorch_lightning as pl
from argparse import Namespace


# YOUR CODE STARTS
def calculate_bleu(lightning_model: pl.LightningModule, config: Namespace) -> float:
    """
    Function that compute bleu score of your lightning model
    STEPS:
    1. Load english and russian validation data
    2. Generate russian translation with lightning model for all english source texts and turns it to texts
    3. Tokenize generated russian texts and validation russian texts
    4. Compute bleu with nltk
    :param lightning_model: trained model with lightning interface
    :param config: hyper parameters of your experiment
    :return: bleu score of your trained model
    """

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
