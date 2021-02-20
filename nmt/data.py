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

from typing import List

from torch.utils.data import Dataset


def load_file(file_path: str) -> List[str]:
    """
    Just loader for file with lines
    :param file_path: path to file
    :return: list of lines of your data
    """

    data: List[str] = list()

    with open(file_path) as file_object:
        for line in file_object:
            data.append(line.strip())

    return data


class Sequence2SequenceDataset(Dataset):

    def __init__(self,
                 source_language_data: List[str],
                 target_language_data: List[str]):
        """
        A Dataset object to handle seq2seq data
        :param source_language_data: list of source texts
        :param target_language_data: list of target texts
        """

        self.source_language_data = source_language_data
        self.target_language_data = target_language_data

    def __len__(self):
        """
        Need for Loader
        :return: len of your dataset
        """
        return len(self.source_language_data)

    def __getitem__(self, index: int) -> (str, str):
        """
        Turn idx to train sample
        :param index: index of your data
        :return: two strings of source and target texts
        """

        source_text = self.source_language_data[index]
        target_text = self.target_language_data[index]

        return source_text, target_text
