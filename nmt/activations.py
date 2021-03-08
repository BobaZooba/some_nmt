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
from typing import Optional

import torch
from torch import nn


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return swish(x)


class GELU(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return gelu(x)


ACTIVATIONS_MAPPER = {
    'relu': nn.ReLU(),
    'swish': Swish(),
    'gelu': GELU()
}


def get_activation_function(activation: Optional[str] = None):

    activation_function = ACTIVATIONS_MAPPER.get(activation, nn.Identity())

    return activation_function
