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

import logging
import os
import random
import zipfile

import requests
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
from tqdm import tqdm

EN_RU_OPEN_SUBTITLES_URL: str = 'http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-ru.txt.zip'
EN_OPEN_SUBTITLES_FILE: str = 'OpenSubtitles.en-ru.en'
RU_OPEN_SUBTITLES_FILE: str = 'OpenSubtitles.en-ru.ru'
TEXT_NORMALIZER: Sequence = Sequence([NFD(), Lowercase(), StripAccents()])

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def download_file(url: str, save_path: str, verbose: bool = False):
    """
    Download data from some url
    :param url: url of your data
    :param save_path: where we need save result
    :param verbose: progress bar
    :return: None
    """
    try:
        filename = save_path.split('/')[-1]
        with requests.get(url, stream=True) as req:
            req.raise_for_status()
            with open(save_path, 'wb') as file_object:
                for chunk in tqdm(req.iter_content(chunk_size=8192),
                                  desc=f'Download {filename}',
                                  disable=not verbose):
                    if chunk:
                        file_object.write(chunk)
    except KeyboardInterrupt as exception:
        os.remove(save_path)
        raise exception
    except Exception as exception:
        os.remove(save_path)
        raise exception


def clean_text(text: str) -> str:
    """
    Normalization of your text sample
    :param text: string with text
    :return: cleaned text
    """
    text = TEXT_NORMALIZER.normalize_str(text)

    if text.startswith('- '):
        text = text[2:]
    elif text.startswith('-'):
        text = text[1:]

    return text


def load_open_subtitles(directory: str,
                        download: bool = True,
                        train_n_pairs: int = 5_000_000,
                        valid_n_pairs: int = 25_000,
                        verbose: bool = False):
    """
    Loading your data from directory and turn it into train and validation files
    :param directory: directory where we can find data
    :param download: do we need to download data
    :param train_n_pairs: how many pairs we need for training
    :param valid_n_pairs: how many pairs we need for validation
    :param verbose: progress bar and logging
    :return: None
    """

    url: str = EN_RU_OPEN_SUBTITLES_URL

    file_name = url.split('/')[-1]

    archive_path: str = os.path.join(directory, file_name)

    if download:
        download_file(url, archive_path, verbose)
        logger.info('Download complete')

    logger.info('Reading files')
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        english_texts = zip_ref.read(EN_OPEN_SUBTITLES_FILE).decode().split('\n')
        russian_texts = zip_ref.read(RU_OPEN_SUBTITLES_FILE).decode().split('\n')
    logger.info('Reading files complete')

    indices = list(range(len(english_texts)))
    random.shuffle(indices)

    english_train_file = open(os.path.join(directory, 'train_en.txt'), 'w')
    russian_train_file = open(os.path.join(directory, 'train_ru.txt'), 'w')

    english_valid_file = open(os.path.join(directory, 'valid_en.txt'), 'w')
    russian_valid_file = open(os.path.join(directory, 'valid_ru.txt'), 'w')

    train_counter: int = 0
    valid_counter: int = 0

    progress_bar = tqdm(total=train_n_pairs + valid_n_pairs, desc='Reading data', disable=not verbose)

    for n, index in enumerate(indices):

        if valid_counter < valid_n_pairs:
            en_file, ru_file = english_valid_file, russian_valid_file
            valid_counter += 1
        elif train_counter < train_n_pairs:
            en_file, ru_file = english_train_file, russian_train_file
            train_counter += 1
        else:
            break

        en_file.write(clean_text(english_texts[index]) + '\n')
        ru_file.write(clean_text(russian_texts[index]) + '\n')

        progress_bar.update()

    progress_bar.close()

    english_train_file.close()
    russian_train_file.close()
    english_valid_file.close()
    russian_valid_file.close()
    logger.info('Open subtitles loaded!')
