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
import torch
import random
import os
import numpy as np
from argparse import ArgumentParser, Namespace
from nmt.get_data import load_open_subtitles
from nmt.tokenizer import train_seq2seq_tokenizers
import pytorch_lightning as pl
from nmt import lightning, metrics
from pytorch_lightning.loggers import WandbLogger


def set_global_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args() -> Namespace:

    parser = ArgumentParser(add_help=False)

    parser.add_argument('--directory', type=str, default='./data/')
    parser.add_argument('--checkpoint_path', type=str, default='./data/checkpoints/checkpoint')
    parser.add_argument('--project_name', type=str, default='NMT')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--train_tokenizers', action='store_true')

    parser.add_argument('--gpu', type=int, default=1 if torch.cuda.is_available() else 0)

    parser.add_argument('--max_norm', type=float, default=3.)

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=32)

    parser.add_argument('--n_batch_accumulate', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_n_pairs', type=int, default=5_000_000)
    parser.add_argument('--valid_n_pairs', type=int, default=25_000)

    parser.add_argument('--pad_index', type=int, default=0)
    parser.add_argument('--bos_index', type=int, default=1)
    parser.add_argument('--eos_index', type=int, default=2)

    parser.add_argument('--vocab_size', type=int, default=30_000)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--encoder_num_layers', type=int, default=2)
    parser.add_argument('--decoder_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--weight_tying', action='store_true')
    parser.add_argument('--use_attention', action='store_true')

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':

    logger = logging.getLogger(__file__)

    args = get_args()

    set_global_seed(args.seed)

    logging.basicConfig(level=logging.INFO)

    if args.load_data:
        logger.info('Loading data')
        try:
            os.mkdir(path=args.directory)
        except FileExistsError:
            pass
        load_open_subtitles(directory=args.directory,
                            verbose=args.verbose,
                            train_n_pairs=args.train_n_pairs,
                            valid_n_pairs=args.valid_n_pairs)
        logger.info('Data loaded')

    if args.train_tokenizers:
        logger.info('Start seq2seq tokenizers training')
        train_seq2seq_tokenizers(directory=args.directory,
                                 max_length=args.max_length,
                                 vocab_size=args.vocab_size)
        logger.info('Tokenizers trained')

    model = lightning.LightningSequence2Sequence(hparams=args)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filepath=os.path.join(os.getcwd(), args.checkpoint_path),
    #     save_last=True,
    #     verbose=args.verbose,
    #     monitor='val_loss',
    #     mode='min',
    #     prefix='seq2seq'
    # )
    #
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), args.checkpoint_path),
        save_last=True,
        verbose=args.verbose,
        monitor='val_loss',
        mode='min',
        prefix='seq2seq'
    )

    try:
        import apex
        use_amp = True
        precision = 16
    except ModuleNotFoundError:
        use_amp = False
        precision = 32
        logger.info('Train without amp')

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accumulate_grad_batches=args.n_batch_accumulate,
                         amp_backend='apex' if use_amp else 'native',
                         precision=precision,
                         gradient_clip_val=args.max_norm,
                         gpus=args.gpu,
                         val_check_interval=5000,
                         num_sanity_val_steps=0,
                         progress_bar_refresh_rate=10,
                         callbacks=[checkpoint_callback],
                         logger=WandbLogger(project=args.project_name))

    trainer.fit(model)

    bleu_score = metrics.calculate_bleu(lightning_model=model, config=args)

    logger.info(f'BLEU score: {bleu_score:.3f}')
