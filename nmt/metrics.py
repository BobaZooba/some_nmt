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

import math
import os
from argparse import Namespace

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from nmt.data import load_file
from nmt.lightning import LightningSequence2Sequence


# YOUR CODE STARTS
def calculate_bleu(lightning_model: LightningSequence2Sequence, config: Namespace) -> float:
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

    ...

# YOUR CODE ENDS
