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

from argparse import Namespace

TEST_DATA_DIRECTORY = './tests/test_data/'


DEFAULT_CONFIG = Namespace(attention_dropout=0.1,
                           batch_size=64,
                           bidirectional_encoder=False,
                           bos_index=1,
                           checkpoint_path='./data/checkpoints/checkpoint',
                           decoder_num_layers=2,
                           directory='./data/',
                           dropout=0.25,
                           embedding_dim=128,
                           encoder_num_layers=2,
                           eos_index=2,
                           epochs=3,
                           gpu=1,
                           learning_rate=0.001,
                           load_data=False,
                           max_length=32,
                           max_norm=3.0,
                           model_dim=256,
                           n_batch_accumulate=1,
                           pad_index=0,
                           project_name='NMT',
                           seed=42,
                           train_n_pairs=5000000,
                           train_tokenizers=False,
                           use_attention=False,
                           valid_n_pairs=25000,
                           verbose=False,
                           vocab_size=30000,
                           weight_decay=0.0,
                           weight_tying=False)
