import os
import random
import zipfile

import requests
from tokenizers.normalizers import Sequence, Lowercase, NFD, StripAccents
from tqdm import tqdm
import logging

EN_RU_OPEN_SUBTITLES_URL: str = 'http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-ru.txt.zip'
TEXT_NORMALIZER: Sequence = Sequence([NFD(), Lowercase(), StripAccents()])

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def download_file(url: str, save_path: str, verbose: bool = False):
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
    text = TEXT_NORMALIZER.normalize_str(text)

    if text.startswith('- '):
        text = text[2:]
    elif text.startswith('-'):
        text = text[1:]

    return text


def load_open_subtitles(directory: str,
                        download: bool = True,
                        verbose: bool = False,
                        valid_n_pairs: int = 25_000):

    url: str = EN_RU_OPEN_SUBTITLES_URL

    file_name = url.split('/')[-1]

    archive_path: str = os.path.join(directory, file_name)

    if download:
        download_file(url, archive_path, verbose)
        logger.info('Download complete')

    logger.info('Reading files')
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        english_texts = zip_ref.read('OpenSubtitles.en-ru.en').decode().split('\n')
        russian_texts = zip_ref.read('OpenSubtitles.en-ru.ru').decode().split('\n')
    logger.info('Reading files complete')

    indices = list(range(len(english_texts)))
    random.shuffle(indices)

    english_train_file = open(os.path.join(directory, 'train_en.txt'), 'w')
    russian_train_file = open(os.path.join(directory, 'train_ru.txt'), 'w')

    english_valid_file = open(os.path.join(directory, 'valid_en.txt'), 'w')
    russian_valid_file = open(os.path.join(directory, 'valid_ru.txt'), 'w')

    for n, index in enumerate(tqdm(indices, desc='Reading data')):

        if len(english_texts) - n <= valid_n_pairs:
            en_file, ru_file = english_valid_file, russian_valid_file
        else:
            en_file, ru_file = english_train_file, russian_train_file

        en_file.write(clean_text(english_texts[index]) + '\n')
        ru_file.write(clean_text(russian_texts[index]) + '\n')

    english_train_file.close()
    russian_train_file.close()
    english_valid_file.close()
    russian_valid_file.close()
    logger.info('Open subtitles loaded!')
