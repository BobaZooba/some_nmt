import logging
import torch
import random
import os
import numpy as np
from argparse import ArgumentParser, Namespace
from get_data import load_open_subtitles
from src.tokenizer import train_seq2seq_tokenizers
import pytorch_lightning as pl
from src import lightning


def set_global_seed(seed: int):
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
    parser.add_argument('--checkpoint_path', type=str, default='./data/amazon/checkpoint')
    parser.add_argument('--project_name', type=str, default='LightningConversation')

    parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0)

    parser.add_argument('--max_norm', type=float, default=3.)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=30_000)

    parser.add_argument('--n_batch_accumulate', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--valid_n_pairs', type=int, default=25_000)

    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':

    logger = logging.getLogger(__file__)

    args = get_args()

    set_global_seed(args.seed)

    logging.basicConfig(level=logging.INFO)

    logger.info('Loading data')
    load_open_subtitles(directory=args.directory,
                        verbose=args.verbose,
                        valid_n_pairs=args.valid_n_pairs)
    logger.info('Data loaded')

    logger.info('Start seq2seq tokenizers training')
    train_seq2seq_tokenizers(directory=args.directory,
                             max_length=args.max_length,
                             vocab_size=args.vocab_size)
    logger.info('Tokenizers trained')

    model = lightning.LightningSequence2Sequence(hparams=args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), args.checkpoint_path),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=args.model_type
    )

    try:
        import apex
        use_amp = True
        precision = 16
    except ModuleNotFoundError:
        use_amp = False
        precision = 32
        logger.info('Train without amp, you can install it with command: make install-apex')

    trainer = pl.Trainer(max_epochs=args.epochs,
                         # logger=pl_logger,
                         accumulate_grad_batches=args.n_batch_accumulate,
                         use_amp=use_amp,
                         precision=precision,
                         gradient_clip_val=args.max_norm,
                         gpus=args.gpus,
                         val_check_interval=5000,
                         num_sanity_val_steps=0,
                         log_save_interval=10,
                         progress_bar_refresh_rate=10,
                         checkpoint_callback=checkpoint_callback)

    trainer.fit(model)
