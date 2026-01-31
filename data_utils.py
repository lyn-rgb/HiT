# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
from bisect import bisect_right
from glob import glob
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    pq = None
    _PARQUET_IMPORT_ERROR = exc
else:
    _PARQUET_IMPORT_ERROR = None


def _resolve_parquet_files(path: str, subset: Optional[str] = None) -> List[str]:
    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "*.parquet")))
    else:
        files = sorted(glob(path))
        if not files and os.path.isfile(path) and path.endswith(".parquet"):
            files = [path]
    if subset is not None:
        subset = subset.lower()
        files = [
            f for f in files
            if os.path.basename(f).lower().startswith(f"{subset}")
        ]
    if not files:
        if subset is None:
            raise FileNotFoundError(f"No parquet files found at {path}")
        raise FileNotFoundError(f"No parquet files found at {path} for subset '{subset}'")
    return files


class ParquetImageDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform=None,
        image_key: str = "image",
        label_key: str = "label",
        subset: Optional[str] = None,
        use_threads: bool = True,
        memory_map: bool = False,
        pre_buffer: bool = False,
        buffer_size: int = 0,
    ):
        if _PARQUET_IMPORT_ERROR is not None:
            raise ModuleNotFoundError("pyarrow is required for parquet datasets") from _PARQUET_IMPORT_ERROR

        if subset is not None:
            subset = subset.lower()
            if subset not in {"train", "test", "validation", "val"}:
                raise ValueError(f"subset must be one of train, test, validation, val; got {subset}")
            if subset == "val":
                subset = "validation"
        self.files = _resolve_parquet_files(path, subset=subset)
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.use_threads = use_threads
        self.memory_map = memory_map
        self.pre_buffer = pre_buffer
        self.buffer_size = buffer_size

        self._file_row_group_counts: List[List[int]] = []
        self._file_row_group_offsets: List[List[int]] = []
        self._file_total_rows: List[int] = []

        for file_path in self.files:
            pf = pq.ParquetFile(
                file_path,
                memory_map=self.memory_map,
                pre_buffer=self.pre_buffer,
                buffer_size=self.buffer_size,
            )
            row_group_counts = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
            offsets = []
            running = 0
            for count in row_group_counts:
                running += count
                offsets.append(running)
            self._file_row_group_counts.append(row_group_counts)
            self._file_row_group_offsets.append(offsets)
            self._file_total_rows.append(running)

        self._cum_file_rows: List[int] = []
        total = 0
        for rows in self._file_total_rows:
            total += rows
            self._cum_file_rows.append(total)

        self._cache_file_idx: Optional[int] = None
        self._cache_row_group_idx: Optional[int] = None
        self._cache_cols: Optional[dict] = None
        self._parquet_file_cache: dict = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop file handles and cached columns to keep the dataset picklable.
        state["_parquet_file_cache"] = {}
        state["_cache_file_idx"] = None
        state["_cache_row_group_idx"] = None
        state["_cache_cols"] = None
        return state

    def __len__(self) -> int:
        return self._cum_file_rows[-1] if self._cum_file_rows else 0

    def _load_row_group(self, file_idx: int, row_group_idx: int) -> dict:
        if (
            self._cache_file_idx == file_idx
            and self._cache_row_group_idx == row_group_idx
            and self._cache_cols is not None
        ):
            return self._cache_cols

        pf = self._parquet_file_cache.get(file_idx)
        if pf is None:
            pf = pq.ParquetFile(
                self.files[file_idx],
                memory_map=self.memory_map,
                pre_buffer=self.pre_buffer,
                buffer_size=self.buffer_size,
            )
            self._parquet_file_cache[file_idx] = pf
        table = pf.read_row_group(
            row_group_idx,
            columns=[self.image_key, self.label_key],
            use_threads=self.use_threads,
        )
        cols = table.to_pydict()
        self._cache_file_idx = file_idx
        self._cache_row_group_idx = row_group_idx
        self._cache_cols = cols
        return cols

    def _resolve_index(self, idx: int) -> Tuple[int, int, int]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        file_idx = bisect_right(self._cum_file_rows, idx)
        file_start = 0 if file_idx == 0 else self._cum_file_rows[file_idx - 1]
        local_idx = idx - file_start

        row_group_offsets = self._file_row_group_offsets[file_idx]
        row_group_idx = bisect_right(row_group_offsets, local_idx)
        row_group_start = 0 if row_group_idx == 0 else row_group_offsets[row_group_idx - 1]
        row_idx = local_idx - row_group_start

        return file_idx, row_group_idx, row_idx

    def _decode_image(self, value) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, dict):
            if "bytes" in value and value["bytes"] is not None:
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            if "path" in value and value["path"]:
                return Image.open(value["path"]).convert("RGB")
        if isinstance(value, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(bytes(value))).convert("RGB")
        if hasattr(value, "to_pylist"):
            value = value.to_pylist()
        if hasattr(value, "tolist"):
            import numpy as np
            arr = np.asarray(value)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(value)}")

    def __getitem__(self, idx: int):
        file_idx, row_group_idx, row_idx = self._resolve_index(idx)
        cols = self._load_row_group(file_idx, row_group_idx)

        if self.image_key not in cols:
            raise KeyError(f"Missing image column '{self.image_key}'")
        if self.label_key not in cols:
            raise KeyError(f"Missing label column '{self.label_key}'")

        image = self._decode_image(cols[self.image_key][row_idx])
        label = cols[self.label_key][row_idx]

        if self.transform is not None:
            image = self.transform(image)

        if isinstance(label, torch.Tensor):
            label = label.item()
        return image, int(label)
