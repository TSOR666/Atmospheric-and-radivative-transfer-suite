"""
Data loading utilities for the refactored STRATUS framework.

This module replaces the ad-hoc dataset implementations that lived in the legacy
`stratus_v4*_utilities.py` scripts.  The new `StratusDataset` exposes a clean and
PyTorch-friendly API while supporting the mixed file formats relied upon by the
project (HDF5, NetCDF, and NumPy archives).  Lightweight LRU caching is used to
avoid redundant disk reads when the same sample is requested repeatedly, and all
optional dependencies degrade gracefully.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    get_worker_info,
)

from .config import StratusConfig

logger = logging.getLogger("stratus.data")

# Optional heavy dependencies -------------------------------------------------
try:  # pragma: no cover - optional dependency
    import h5py  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    h5py = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import netCDF4 as nc  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    nc = None  # type: ignore


SupportedPath = Union[str, Path]
Sample = dict[str, torch.Tensor]


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")


def _optional_import_warning(package: str, ext: str) -> None:
    logger.warning(
        "Skipping file with extension %s because dependency '%s' is not installed.",
        ext,
        package,
    )


class StratusDataset(Dataset):
    """
    Dataset loader capable of ingesting heterogeneous STRATUS datasets.

    Parameters
    ----------
    data_path:
        Root directory containing the dataset files.
    config:
        STRATUS configuration describing grid/canonical shapes.
    transform:
        Optional callable applied to each sample before returning.
    cache_size:
        Number of recently accessed samples kept in memory.  Set to zero to disable.
    """

    SUPPORTED_SUFFIXES: tuple[str, ...] = (".h5", ".hdf5", ".nc", ".nc4", ".npz")

    def __init__(
        self,
        data_path: SupportedPath,
        config: StratusConfig,
        transform: Callable[[Sample], Sample] | None = None,
        cache_size: int = 64,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.config = config
        self.transform = transform
        self.cache_size = max(0, cache_size)
        self._cache: OrderedDict[int, Sample] = OrderedDict()
        self._cache_disabled_for_workers = False

        _ensure_exists(self.data_path)

        self.metadata = self._load_metadata()
        self.files = self._discover_files()

        if not self.files:
            raise RuntimeError(f"No dataset files found under {self.data_path}.")

        logger.info(
            "Loaded STRATUS dataset metadata: %s samples, formats: %s",
            len(self),
            sorted({path.suffix for path in self.files}),
        )

    # ------------------------------------------------------------------ helpers
    def _load_metadata(self) -> dict[str, int | str | tuple[int, ...] | None]:
        meta_file = self.data_path / "metadata.json"
        if meta_file.exists():
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                return metadata
            except Exception as exc:
                logger.warning("Failed to parse metadata.json: %s", exc)

        return {
            "num_samples": None,
            "grid_shape": tuple(self.config.grid_shape),
            "n_bands": self.config.n_bands,
            "n_stokes": self.config.n_stokes,
        }

    def _discover_files(self) -> list[Path]:
        files: list[Path] = []
        root = self.data_path.resolve()
        for ext in self.SUPPORTED_SUFFIXES:
            for path in self.data_path.glob(f"**/*{ext}"):
                try:
                    resolved = path.resolve(strict=True)
                except FileNotFoundError:
                    logger.warning("Skipping disappearing file during discovery: %s", path)
                    continue
                if not resolved.is_relative_to(root):
                    logger.warning(
                        "Skipping file outside dataset root (possible symlink traversal): %s",
                        resolved,
                    )
                    continue
                files.append(resolved)
        files.sort()

        num_samples = self.metadata.get("num_samples")
        if isinstance(num_samples, int) and num_samples != len(files):
            logger.debug(
                "Metadata reported %s samples but %s files were found.",
                num_samples,
                len(files),
            )
        return files

    # ---------------------------------------------------------------- interface
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Sample:
        if index < 0 or index >= len(self.files):
            raise IndexError(index)

        cached = self._cache_get(index)
        if cached is not None:
            return cached

        path = self.files[index]
        sample = self._read_file(path)
        if "kappa" not in sample or "source" not in sample:
            raise ValueError(f"Sample {path} is missing required 'kappa' or 'source' data.")

        if self.transform is not None:
            sample = self.transform(sample)

        self._cache_set(index, sample)
        return sample

    # ---------------------------------------------------------------- utilities
    def _cache_get(self, index: int) -> Sample | None:
        if not self._cache_available():
            return None
        try:
            sample = self._cache.pop(index)
        except KeyError:
            return None
        # Reinsert to mark as most recently used
        self._cache[index] = sample
        return sample

    def _cache_set(self, index: int, sample: Sample) -> None:
        if not self._cache_available():
            return
        if index in self._cache:
            self._cache.pop(index)
        elif len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[index] = sample

    def _cache_available(self) -> bool:
        if self.cache_size == 0:
            return False
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1:
            if not self._cache_disabled_for_workers:
                logger.warning(
                    "Disabling dataset cache because DataLoader is using %s workers.",
                    worker.num_workers,
                )
                self._cache_disabled_for_workers = True
            self._cache.clear()
            return False
        return True

    def _read_file(self, path: Path) -> Sample:
        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            if h5py is None:  # pragma: no cover - optional dependency
                _optional_import_warning("h5py", suffix)
                return {}
            return self._load_hdf5(path)
        if suffix in {".nc", ".nc4"}:
            if nc is None:  # pragma: no cover - optional dependency
                _optional_import_warning("netCDF4", suffix)
                return {}
            return self._load_netcdf(path)
        if suffix == ".npz":
            return self._load_npz(path)

        raise ValueError(f"Unsupported file extension: {suffix}")

    def _load_hdf5(self, path: Path) -> Sample:
        sample: Sample = {}
        with h5py.File(path, "r") as handle:  # type: ignore[arg-type]
            for key in ("kappa", "source", "radiance", "evaluation_points"):
                if key not in handle:
                    continue
                data = handle[key][:]
                tensor = torch.from_numpy(np.asarray(data)).float()
                sample[key if key != "radiance" else "targets"] = tensor

        if not sample:
            logger.warning("No recognised datasets in %s.", path)
        return sample

    def _load_netcdf(self, path: Path) -> Sample:
        sample: Sample = {}
        with nc.Dataset(path, "r") as handle:  # type: ignore[arg-type]
            mapping = {
                "extinction": "kappa",
                "emission": "source",
                "radiance": "targets",
            }
            for source_name, target_key in mapping.items():
                if source_name in handle.variables:
                    array = handle.variables[source_name][:]  # type: ignore[index]
                    sample[target_key] = torch.from_numpy(np.asarray(array)).float()

            if "evaluation_points" in handle.variables:
                array = handle.variables["evaluation_points"][:]  # type: ignore[index]
                sample["evaluation_points"] = torch.from_numpy(np.asarray(array)).float()

        if not sample:
            logger.warning("No recognised variables in %s.", path)
        return sample

    def _load_npz(self, path: Path) -> Sample:
        sample: Sample = {}
        with np.load(path) as data:  # type: ignore[arg-type]
            for key in ("kappa", "source", "radiance", "evaluation_points"):
                if key not in data:
                    continue
                tensor = torch.from_numpy(np.asarray(data[key])).float()
                sample[key if key != "radiance" else "targets"] = tensor
        if not sample:
            logger.warning("No recognised arrays in %s.", path)
        return sample

    # ---------------------------------------------------------------- lifecycle
    def clear_cache(self) -> None:
        """Drop all cached samples to release memory."""
        self._cache.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.clear_cache()
        except Exception:
            pass


def create_dataloader(
    data_path: SupportedPath,
    config: StratusConfig,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    transform: Callable[[Sample], Sample] | None = None,
    cache_size: int = 64,
    distributed: bool = False,
) -> tuple[DataLoader, StratusDataset, DistributedSampler | None]:
    """
    Convenience helper to instantiate a :class:`StratusDataset` and wrap it in a
    :class:`torch.utils.data.DataLoader`.
    """

    dataset = StratusDataset(
        data_path=data_path,
        config=config,
        transform=transform,
        cache_size=cache_size,
    )

    sampler: DistributedSampler[Sample] | None = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Sampler handles shuffling.

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, dataset, sampler


__all__ = ["StratusDataset", "create_dataloader"]
