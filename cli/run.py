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
import sys

import nltk
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from nmt import lightning, metrics, utils
from nmt.get_data import load_open_subtitles
from nmt.tokenizer import train_seq2seq_tokenizers

if __name__ == '__main__':

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    nltk.download('punkt')

    logger = logging.getLogger(os.path.basename(__file__))

    args = utils.get_args()

    pl.seed_everything(seed=args.seed)

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

    try:
        import apex
        use_amp = True
        precision = 16
    except ModuleNotFoundError:
        use_amp = False
        precision = 32
        logger.info('Train without amp')

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), args.checkpoint_path),
        save_last=True,
        verbose=args.verbose,
        monitor='val_loss',
        mode='min'
    )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accumulate_grad_batches=args.n_batch_accumulate,
                         amp_backend='apex' if use_amp else 'native',
                         precision=precision,
                         gradient_clip_val=args.max_norm,
                         gpus=args.gpu,
                         val_check_interval=args.val_check_interval,
                         num_sanity_val_steps=0,
                         callbacks=[checkpoint_callback],
                         logger=WandbLogger(project=args.project_name))

    trainer.fit(model)

    torch.save(model.state_dict(), args.state_dict_path)

    bleu_score = metrics.calculate_bleu(lightning_model=model, config=args)

    logger.info(f'BLEU score: {bleu_score:.3f}')
