"""
Load independent EEG splits as disk batches or normalized BF16 RAM caches.

Copyright 2022 Centre for Brain Computing Research (CBCR), College of Computing and Data Science (CCDS), Nanyang Technological University (NTU);
licensed under the CBCR License 1.0 (see LICENSE).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

from torch.utils.data import BatchSampler, DataLoader, Dataset
import h5py
import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class SampleRef:
    dataset: str
    path: Path
    x_key: str
    row: int
    local_y: int
    sample_id: str


class RAMSplit(Dataset):
    """In-memory split for small synthetic tests and benchmarks."""

    def __init__(self, x: np.ndarray, y: np.ndarray, source: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.source = torch.from_numpy(source)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.source[index]


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Temporal z-score for every trial and channel."""
    mean = x.mean(-1, keepdim=True)
    scale = x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (x - mean) / scale


def prepare_input(x, split, device, dtype):
    """Cached inputs are already normalized; disk inputs still need normalization."""
    x = x.to(device, non_blocking=True)
    if not getattr(split, "normalized", False):
        x = normalize(x)
    return x.to(dtype)


class BF16RAMSplit(RAMSplit):
    """Process-local normalized EEG; no disk handles or FP32 EEG are retained."""

    normalized = True

    @torch.no_grad()
    def __init__(self, disk, device, batch_size):
        self.y, self.source = disk.y, disk.source
        self.x = torch.empty((len(disk), *disk.input_shape), dtype=torch.bfloat16)
        try:
            for start in range(0, len(disk), batch_size):
                end = min(start + batch_size, len(disk))
                x, _, _ = disk[list(range(start, end))]
                self.x[start:end] = normalize(x.to(device)).to(torch.bfloat16).cpu()
        finally:
            disk.close()


class DiskSplit(Dataset):
    """Read one batch at a time; Linux caches file pages across processes."""

    def __init__(self, refs, mapping, source_ids, input_shape):
        self.refs = refs
        self.input_shape = input_shape
        self.y = torch.tensor([mapping[r.dataset, r.local_y] for r in refs])
        self.source = torch.tensor([source_ids[r.dataset] for r in refs])
        self._files = {}

    def __len__(self):
        return len(self.refs)

    def __getstate__(self):
        # Spawned loader workers open their own HDF5 handles.
        return {**self.__dict__, "_files": {}}

    def __getitem__(self, indices):
        single = isinstance(indices, int)
        indices = [indices] if single else list(indices)
        x = np.empty((len(indices), *self.input_shape), dtype=np.float32)
        groups = {}
        for position, index in enumerate(indices):
            ref = self.refs[index]
            groups.setdefault((ref.path, ref.x_key), []).append((position, ref.row))
        for (path, key), entries in groups.items():
            if path not in self._files:
                self._files[path] = h5py.File(path, "r", locking=False)
            positions, rows = zip(*entries)
            # h5py requires increasing, unique row indices. Restore batch order afterward.
            unique, inverse = np.unique(rows, return_inverse=True)
            dataset = self._files[path][key]
            target = self.input_shape[1]
            samples = dataset.shape[-1]
            start = max(0, (samples - target) // 2)
            values = np.asarray(dataset[unique, :, start:start + target], dtype=np.float32)
            if samples < target:
                left = (target - samples) // 2
                values = np.pad(values, ((0, 0), (0, 0), (left, target - samples - left)))
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite EEG in {path}:{key}")
            x[list(positions)] = values[inverse]
        batch = torch.from_numpy(x), self.y[indices], self.source[indices]
        return tuple(value[0] for value in batch) if single else batch

    def close(self):
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def __del__(self):
        self.close()


class SplitBatches(Dataset):
    """Route (split name, batch indices) requests through one worker pool."""

    def __init__(self, splits):
        self.splits = splits
        self.normalized = all(getattr(split, "normalized", False) for split in splits.values())

    def __getitem__(self, request):
        name, indices = request
        return self.splits[name][indices]


def batch_loader(split, batch_size, device, *, num_workers=0, prefetch_factor=2, batches=None):
    """Prefetch bounded batches, with RNG independent of model and shuffle RNG."""
    # RAM indexing runs in the trainer: never send the full cache to worker SHM.
    if getattr(split, "normalized", False):
        num_workers = 0
    sampler = BatchSampler(range(len(split)), batch_size, drop_last=False) if batches is None else batches
    options = {}
    if num_workers:
        options.update(
            multiprocessing_context="spawn", persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(
        split, batch_size=None, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.device(device).type == "cuda",
        generator=torch.Generator().manual_seed(0), **options,
    )


class CombinedData:
    """Describe split membership, then open only explicitly requested disk datasets."""

    def __init__(self, config: dict):
        self.config = config
        settings = config["data"]
        catalog_path = Path(settings["catalog"])
        catalog = yaml.safe_load(catalog_path.read_text())
        self.catalog = catalog["datasets"]
        self.names = list(self.catalog)
        self.input_shape = tuple(catalog["input_shape"])
        self.validation_fraction = float(catalog["validation_fraction"])
        self.refs = {split: [] for split in ("train", "valid", "test")}
        self.splits: dict[str, DiskSplit] = {}

        root_value = catalog.get("path")
        root_base = catalog_path.parent
        paths_file = catalog_path.parent.parent / "path.yaml"
        if root_value is None and paths_file.is_file():
            # Catalog names (emotion, mi, sleep) select their shared directory.
            root_value = yaml.safe_load(paths_file.read_text())[catalog_path.stem]
            root_base = paths_file.parent
        root = Path(root_value if root_value is not None else ".").expanduser()
        if not root.is_absolute():
            root = root_base / root
        root = root.resolve()
        self.report = {
            "input_shape": list(self.input_shape),
            "dtype": "float32",
            "datasets": {},
        }
        for name in self.names:
            spec = self.catalog[name]
            path = Path(spec["filename"]).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Dataset file not found: {path}")
            split_refs = {
                split: self._build_refs(name, spec, path, split)
                for split in self.refs
            }
            seen: set[tuple[str, int]] = set()
            hashes = {}
            for split, refs in split_refs.items():
                identities = [(ref.x_key, ref.row) for ref in refs]
                if not refs or len(identities) != len(set(identities)):
                    raise ValueError(f"Empty or duplicate split: {name}/{split}")
                if seen.intersection(identities):
                    raise ValueError(f"Overlapping split: {name}/{split}")
                seen.update(identities)
                hashes[split] = hashlib.sha256(
                    json.dumps(identities).encode()
                ).hexdigest()
                self.refs[split].extend(refs)
            stat = path.stat()
            self.report["datasets"][name] = {
                "path": str(path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "counts": {split: len(refs) for split, refs in split_refs.items()},
                "sample_identity_sha256": hashes,
            }

        self.report["counts"] = {
            split: len(refs) for split, refs in self.refs.items()
        }
        bytes_per_trial = int(np.prod(self.input_shape)) * 4 + 16
        self.report["pool_bytes"] = {
            split: len(refs) * bytes_per_trial for split, refs in self.refs.items()
        }
        self.define_labels()

    def _allowed_labels(self, spec: dict) -> list[int]:
        return list(spec.get("include_local_labels", range(len(spec["labels"]))))

    def _check_label(self, name: str, spec: dict, value: int, where: str) -> None:
        if not 0 <= value < len(spec["labels"]):
            raise ValueError(f"{name}: invalid label {value} at {where}")

    def _predefined_refs(self, name: str, spec: dict, path: Path, split: str):
        x_key, y_key = f"{split}X", f"{split}Y"
        refs = []
        with h5py.File(path, "r", locking=False) as h5:
            x, y = h5[x_key], h5[y_key]
            expected = (self.input_shape[0], int(spec["input_samples"]))
            if tuple(x.shape[1:]) != expected or len(x) != len(y):
                raise ValueError(f"Unexpected X/Y shape in {path}:{split}")
            allowed = set(self._allowed_labels(spec))
            for row, raw_y in enumerate(np.asarray(y[:], dtype=np.int64)):
                local_y = int(raw_y)
                self._check_label(name, spec, local_y, f"{y_key}[{row}]")
                if local_y in allowed:
                    refs.append(SampleRef(name, path, x_key, row, local_y, f"{split}:{row}"))
        return refs

    def _subject_refs(self, name: str, spec: dict, path: Path, split: str):
        refs = []
        saved_splits = spec["layout"] == "subject_splits"
        allowed = set(self._allowed_labels(spec))
        with h5py.File(path, "r", locking=False) as h5:
            subjects = list(h5.keys())
            if saved_splits:
                if "split_manifest" in spec:
                    metadata = yaml.safe_load((path.parent / spec["split_manifest"]).read_text())
                    if metadata["export"]["sample_shape"] != list(self.input_shape):
                        raise ValueError(f"Manifest input shape differs from catalog: {path}")
                    if metadata["labels"] != spec["labels"] or not metadata["complete"]:
                        raise ValueError(f"Invalid dataset metadata: {path}")
                    manifest = metadata["splits"]
                else:
                    manifest = json.loads(h5.attrs["splits"])
                assigned = [subject for key in ("train", "valid", "test") for subject in manifest[key]]
                if len(assigned) != len(set(assigned)) or set(assigned) != set(subjects):
                    raise ValueError(f"Invalid or overlapping subject splits in {path}")
                selected = manifest[split]
            else:
                train_subjects = int(spec["train_subjects"])
                selected = subjects[train_subjects:] if split == "test" else subjects[:train_subjects]
            validation_step = round(1 / self.validation_fraction)
            filtered_position = 0
            for subject in selected:
                x, y = h5[f"{subject}/X"], h5[f"{subject}/Y"]
                expected = (self.input_shape[0], int(spec["input_samples"]))
                if tuple(x.shape[1:]) != expected or len(x) != len(y):
                    raise ValueError(f"Unexpected X/Y shape in {path}:{subject}")
                for row, raw_y in enumerate(np.asarray(y[:], dtype=np.int64)):
                    local_y = int(raw_y)
                    self._check_label(name, spec, local_y, f"{subject}/Y[{row}]")
                    if local_y not in allowed:
                        continue
                    if saved_splits or split == "test":
                        use = True
                    else:
                        is_validation = filtered_position % validation_step == 0
                        use = is_validation if split == "valid" else not is_validation
                        filtered_position += 1
                    if use:
                        refs.append(SampleRef(
                            name, path, f"{subject}/X", row, local_y, f"{subject}:{row}"
                        ))
        return refs

    def _build_refs(self, name: str, spec: dict, path: Path, split: str):
        if spec["layout"] == "predefined":
            return self._predefined_refs(name, spec, path, split)
        if spec["layout"] in {"subjects", "subject_splits"}:
            return self._subject_refs(name, spec, path, split)
        raise ValueError(f"Unsupported layout for {name}: {spec['layout']}")

    def define_labels(self) -> dict[tuple[str, int], int]:
        policy = self.config["data"]["label_policy"]
        if policy not in ("shared_semantic", "dataset_specific"):
            raise ValueError("label_policy must be shared_semantic or dataset_specific")
        labels: list[str] = []
        mapping = {}
        source_classes = {}
        for name in self.names:
            spec = self.catalog[name]
            source_classes[name] = []
            for local_y in self._allowed_labels(spec):
                prefix = spec["family"] if policy == "shared_semantic" else name
                label = f"{prefix}::{spec['labels'][local_y]}"
                if label not in labels:
                    labels.append(label)
                global_y = labels.index(label)
                mapping[name, local_y] = global_y
                source_classes[name].append(global_y)
        self.report.update(
            label_policy=policy,
            labels=labels,
            source_classes=source_classes,
        )
        return mapping

    def load(self, splits=("train", "valid"), *, device="cpu"):
        """Load only requested splits, using disk batches or a BF16 RAM cache."""
        storage = self.config["data"].get("storage", "disk")
        if storage not in {"disk", "ram_bf16"}:
            raise ValueError(f"Unknown data storage: {storage}")
        if storage == "ram_bf16":
            from .train import configure_cuda

            training = self.config["training"]
            if training["precision"] not in {"bf16", "bf16_full"}:
                raise ValueError("ram_bf16 requires BF16 mixed or full precision")
            torch.set_num_threads(int(training.get("cpu_threads", 1)))
            configure_cuda(training, torch.device(device))
            batch_size = int(self.config["data"]["preload_batch_size"])
            if batch_size < 1:
                raise ValueError("preload_batch_size must be positive")
        mapping = self.define_labels()
        unknown = set(splits) - set(self.refs)
        if unknown:
            raise ValueError(f"Unknown splits: {sorted(unknown)}")
        source_ids = {name: index for index, name in enumerate(self.names)}
        for split in splits:
            if split not in self.splits:
                disk = DiskSplit(
                    self.refs[split], mapping, source_ids, self.input_shape
                )
                self.splits[split] = BF16RAMSplit(disk, device, batch_size) if storage == "ram_bf16" else disk
        return self
