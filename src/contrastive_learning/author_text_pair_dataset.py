import torch
from torch.utils.data import Dataset


class AuthorTextPairsDataset(Dataset[list[torch.Tensor]]):
    def __init__(self, text_vector_pairs: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        Args:
            text_vector_pairs: List of tuples (text_vec1, text_vec2) representing two personas of the same author.
                Each text_vec is a numpy array representing the persona embedding
        """
        self.text_vector_pairs = text_vector_pairs

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        text_vec1, text_vec2 = self.text_vector_pairs[idx]
        return [text_vec1, text_vec2]

    def __len__(self) -> int:
        return len(self.text_vector_pairs)
