from typing import List

from torch.utils.data import Dataset


def load_file(file_path: str) -> List[str]:

    data: List[str] = list()

    with open(file_path) as file_object:
        for line in file_object:
            data.append(line.strip())

    return data


class Sequence2SequenceDataset(Dataset):

    def __init__(self,
                 source_language_data: List[str],
                 target_language_data: List[str]):

        self.source_language_data = source_language_data
        self.target_language_data = target_language_data

    def __len__(self):
        return len(self.source_language_data)

    def __getitem__(self, index: int) -> (str, str):

        source_text = self.source_language_data[index]
        target_text = self.target_language_data[index]

        return source_text, target_text
