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

import torch

from nmt import lightning, metrics, utils

if __name__ == '__main__':

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

    logger = logging.getLogger(os.path.basename(__file__))

    args = utils.get_args()

    model = lightning.LightningSequence2Sequence(hparams=args)

    model.load_state_dict(torch.load(args.state_dict_path))

    bleu_score = metrics.calculate_bleu(lightning_model=model, config=args)

    logger.info(f'BLEU score: {bleu_score:.3f}')