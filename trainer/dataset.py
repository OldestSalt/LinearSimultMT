import tables
import torch
from pathlib import Path


class TranslationDataset(torch.utils.data.Dataset):
    """
    Dataset for tokenized translation data.
    """

    def __init__(self, path: str | Path, lazy: bool = False):
        self.path = str(path)
        self._file = None
        self.lazy = lazy
        self.source_ids = None
        self.target_ids = None
        self.source_mask = None
        self.target_mask = None

        if not lazy:
            with tables.open_file(self.path, mode="r") as file:
                self.source_ids = file.root.source_ids.read()
                self.target_ids = file.root.target_ids.read()
                self.source_mask = file.root.source_mask.read()
                self.target_mask = file.root.target_mask.read()
                self.teacher_top32_ids = file.root.teacher_top32_ids.read()
                self.teacher_top32_logits = file.root.teacher_top32_logits.read()
                self.synth_ids = file.root.synth_ids.read()
                self.synth_mask = file.root.synth_mask.read()
                

        with tables.open_file(self.path, mode="r") as file:
            self.length = file.root.source_ids.shape[0]

    def _lazy_open(self):
        if self._file is None:
            self._file = tables.open_file(self.path, mode="r")
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.lazy:
            file = self._lazy_open()

            return {
                "source_ids": torch.as_tensor(file.root.source_ids[idx], dtype=torch.long),
                "target_ids": torch.as_tensor(file.root.target_ids[idx], dtype=torch.long),
                "source_mask": torch.as_tensor(file.root.source_mask[idx], dtype=bool),
                "target_mask": torch.as_tensor(file.root.target_mask[idx], dtype=bool),
                "teacher_top32_ids": torch.as_tensor(file.root.teacher_top32_ids[idx], dtype=torch.long)[..., :-1, :], # i fucked up and i have saved last token predictions too
                "teacher_top32_logits": torch.as_tensor(file.root.teacher_top32_logits[idx], dtype=torch.float32)[..., :-1, :],
                "synth_ids": torch.as_tensor(file.root.synth_ids[idx], dtype=torch.long),
                "synth_mask": torch.as_tensor(file.root.synth_mask[idx], dtype=bool)
            }
        
        return {
            "source_ids": torch.tensor(self.source_ids[idx], dtype=torch.long),
            "target_ids": torch.tensor(self.target_ids[idx], dtype=torch.long),
            "source_mask": torch.tensor(self.source_mask[idx], dtype=bool),
            "target_mask": torch.tensor(self.target_mask[idx], dtype=bool),
            "teacher_top32_ids": torch.as_tensor(self.teacher_top32_ids[idx], dtype=torch.long)[..., :-1, :],
            "teacher_top32_logits": torch.as_tensor(self.teacher_top32_logits[idx], dtype=torch.float32)[..., :-1, :],
            "synth_ids": torch.as_tensor(self.synth_ids[idx], dtype=torch.long),
            "synth_mask": torch.as_tensor(self.synth_mask[idx], dtype=bool)
        }

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None