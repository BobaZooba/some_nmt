import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch


def set_global_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def get_args() -> Namespace:
#
#     parser = ArgumentParser(add_help=False)
#
#     parser.add_argument('--directory', type=str, default='./data/')
#     parser.add_argument('--checkpoint_path', type=str, default='./data/checkpoints/')
#     parser.add_argument('--state_dict_path', type=str, default='./data/last_state_dict.pt')
#     parser.add_argument('--project_name', type=str, default='NMT')
#
#     parser.add_argument('--verbose', action='store_true')
#     parser.add_argument('--load_data', action='store_true')
#     parser.add_argument('--train_tokenizers', action='store_true')
#
#     parser.add_argument('--gpu', type=int, default=1 if torch.cuda.is_available() else 0)
#
#     parser.add_argument('--max_norm', type=float, default=3.)
#
#     parser.add_argument('--epochs', type=int, default=5)
#     parser.add_argument('--batch_size', type=int, default=256)
#     parser.add_argument('--max_length', type=int, default=32)
#
#     parser.add_argument('--n_batch_accumulate', type=int, default=1)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--train_n_pairs', type=int, default=10_000_000)
#     parser.add_argument('--valid_n_pairs', type=int, default=50_000)
#     parser.add_argument('--val_check_interval', type=int, default=5_000)
#
#     parser.add_argument('--pad_index', type=int, default=0)
#     parser.add_argument('--bos_index', type=int, default=1)
#     parser.add_argument('--eos_index', type=int, default=2)
#
#     parser.add_argument('--vocab_size', type=int, default=30_000)
#     parser.add_argument('--embedding_dim', type=int, default=128)
#     parser.add_argument('--model_dim', type=int, default=256)
#     parser.add_argument('--encoder_num_layers', type=int, default=3)
#     parser.add_argument('--decoder_num_layers', type=int, default=3)
#     parser.add_argument('--dropout', type=float, default=0.25)
#     parser.add_argument('--weight_tying', action='store_true')
#     parser.add_argument('--bidirectional_encoder', action='store_true')
#     parser.add_argument('--attention_dropout', type=float, default=0.1)
#
#     parser.add_argument('--learning_rate', type=float, default=0.001)
#     parser.add_argument('--weight_decay', type=float, default=0.01)
#
#     parsed_args = parser.parse_args()
#
#     return parsed_args


def get_args() -> Namespace:

    parser = ArgumentParser(add_help=False)

    parser.add_argument('--directory', type=str, default='./data/')
    parser.add_argument('--checkpoint_path', type=str, default='./data/checkpoints/')
    parser.add_argument('--state_dict_path', type=str, default='./data/last_state_dict.pt')
    parser.add_argument('--project_name', type=str, default='NMT')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--load_data', action='store_true')
    parser.add_argument('--train_tokenizers', action='store_true')

    parser.add_argument('--gpu', type=int, default=1 if torch.cuda.is_available() else 0)

    parser.add_argument('--max_norm', type=float, default=3.)

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_batch_accumulate', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=32)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_n_pairs', type=int, default=10_000_000)
    parser.add_argument('--valid_n_pairs', type=int, default=50_000)
    parser.add_argument('--val_check_interval', type=int, default=5_000)

    parser.add_argument('--pad_index', type=int, default=0)
    parser.add_argument('--bos_index', type=int, default=1)
    parser.add_argument('--eos_index', type=int, default=2)

    parser.add_argument('--vocab_size', type=int, default=30_000)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--model_dim', type=int, default=768)
    parser.add_argument('--embedding_dropout', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--input_cnn_kernel_size', type=int, default=3)
    parser.add_argument('--activation', type=str, default='swish')
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--feed_forward_dim', type=int, default=1024)
    parser.add_argument('--head_dim', type=int, default=None)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--use_fusion_gate', action='store_true')
    parser.add_argument('--encoder_num_layers', type=int, default=4)
    parser.add_argument('--decoder_num_layers', type=int, default=8)
    parser.add_argument('--weight_tying', action='store_true')

    parser.add_argument('--warmup_steps', type=int, default=5_000)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parsed_args = parser.parse_args()

    return parsed_args
