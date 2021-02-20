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
from argparse import ArgumentParser, Namespace
from nmt import lightning, metrics


def get_args() -> Namespace:

    parser = ArgumentParser(add_help=False)

    parser.add_argument('--directory', type=str, default='./data/')
    parser.add_argument('--checkpoint_path', type=str, default='./data/checkpoints/checkpoint/seq2seq-checkpoint.ckpt')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=32)

    parser.add_argument('--pad_index', type=int, default=0)
    parser.add_argument('--eos_index', type=int, default=2)

    parser.add_argument('--vocab_size', type=int, default=30_000)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--encoder_num_layers', type=int, default=2)
    parser.add_argument('--decoder_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--weight_tying', action='store_true')
    parser.add_argument('--bidirectional_encoder', action='store_true')
    parser.add_argument('--use_attention', action='store_true')

    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':

    logger = logging.getLogger(__file__)

    args = get_args()

    logging.basicConfig(level=logging.INFO)

    model = lightning.LightningSequence2Sequence(hparams=args)

    model.load_from_checkpoint(args.checkpoint_path)

    bleu_score = metrics.calculate_bleu(lightning_model=model, config=args)

    logger.info(f'BLEU score: {bleu_score:.3f}')
