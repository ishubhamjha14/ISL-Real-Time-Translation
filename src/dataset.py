"""
dataset.py
Phase 4: Dataset Loader and Preprocessing Pipeline for Indian Sign Language (ISL) Translation System.

Requirements:
1. Custom PyTorch Dataset (ISLDataset):
   - Load .npy files dynamically given file paths and targets.
   - Support boolean flag `is_train` to toggle data augmentations.
   - Return input tensor (Float) of shape (30, feature_dim) and label tensor (Long).
2. Augmentations on Landmarks (applied only when is_train=True):
   - Random Temporal Subsampling / Speed Variation: Resample frames slightly faster or slower back to 30 frames.
   - Landmark Jittering: Add Gaussian noise (mean=0, std=0.008) preserving zero-padding.
   - Temporal Dropout: Randomly zero-out 1-2 full frames.
3. Stratified Splitting and DataLoader Utility:
   - `create_dataloaders(processed_dir, label_map_path, batch_size=16, val_split=0.15, test_split=0.15, seed=42)`
   - Balanced train/val/test splits per gloss.
   - Returns train_loader, val_loader, test_loader, feature_dim.
4. Unit Test Block:
   - Verifies batch shapes and data types.
"""

import os
import sys

# Ensure execution in conda environment if required packages are missing
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
except ImportError:
    conda_py = r"C:\Users\abhis\anaconda3\anaconda\envs\isl_py310\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        result = subprocess.run([conda_py] + sys.argv)
        sys.exit(result.returncode)
    raise

import json
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np


def trim_landmarks(data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Slices raw MediaPipe Holistic 1662-D landmark features down to essential upper body & hands (226-D):
    - Upper Pose: landmarks 0 to 24 (100 values: x, y, z, visibility) -> indices 0:100.
    - Face: 1404 dense face mesh features completely dropped -> indices 132:1536 dropped.
    - Left Hand: all 21 keypoints (63 values: x, y, z) -> indices 1536:1599.
    - Right Hand: all 21 keypoints (63 values: x, y, z) -> indices 1599:1662.
    Total output dimension: 100 + 63 + 63 = 226 features.
    """
    if isinstance(data, np.ndarray):
        if data.shape[-1] == 1662:
            return np.concatenate([data[..., 0:100], data[..., 1536:1662]], axis=-1)
        return data
    elif isinstance(data, torch.Tensor):
        if data.size(-1) == 1662:
            return torch.cat([data[..., 0:100], data[..., 1536:1662]], dim=-1)
        return data
    return data


class ISLDataset(Dataset):
    """
    PyTorch Dataset for Indian Sign Language (ISL) Landmark Sequences.

    Args:
        file_paths: List of file paths to .npy sample files.
        labels: List of integer class labels corresponding to file_paths.
        is_train: Boolean flag indicating if data augmentations should be applied.
        fixed_frames: Expected sequence length (default: 30).
        speed_range: Tuple of (min_speed, max_speed) for temporal speed variation.
        jitter_std: Standard deviation of Gaussian noise for landmark jittering.
        max_dropped_frames: Maximum number of frames to drop during temporal dropout.
        trim_features: Boolean flag to trim 1662 features to 226 upper pose & hand features.
    """

    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        labels: List[int],
        is_train: bool = False,
        fixed_frames: int = 30,
        speed_range: Tuple[float, float] = (0.85, 1.15),
        jitter_std: float = 0.008,
        max_dropped_frames: int = 2,
        trim_features: bool = True,
    ):
        assert len(file_paths) == len(labels), "Number of file paths must match number of labels."
        self.file_paths = [Path(p) for p in file_paths]
        self.labels = labels
        self.is_train = is_train
        self.fixed_frames = fixed_frames
        self.speed_range = speed_range
        self.jitter_std = jitter_std
        self.max_dropped_frames = max_dropped_frames
        self.trim_features = trim_features

    def __len__(self) -> int:
        return len(self.file_paths)

    def _apply_dynamic_temporal_warping(
        self,
        tensor: torch.Tensor,
        min_frames: int = 24,
        max_frames: int = 36,
    ) -> torch.Tensor:
        """
        Dynamically resamples sequence between min_frames (24) and max_frames (36)
        and linearly interpolates back to self.fixed_frames (30 frames).
        Simulates signing speed variation (fast vs slow signers).
        """
        target_len = int(np.random.randint(min_frames, max_frames + 1))
        inp = tensor.unsqueeze(0).transpose(1, 2)  # (1, D, T)
        if target_len <= self.fixed_frames:
            max_start = self.fixed_frames - target_len
            start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
            sub = inp[:, :, start : start + target_len]
            warped = F.interpolate(sub, size=self.fixed_frames, mode="linear", align_corners=True)
        else:
            stretched = F.interpolate(inp, size=target_len, mode="linear", align_corners=True)
            max_start = target_len - self.fixed_frames
            start = int(np.random.randint(0, max_start + 1))
            warped = stretched[:, :, start : start + self.fixed_frames]
        return warped.transpose(1, 2).squeeze(0)  # (30, D)

    def _apply_hand_scale_variation(
        self,
        tensor: torch.Tensor,
        scale_range: Tuple[float, float] = (0.92, 1.08),
    ) -> torch.Tensor:
        """
        Scales hand landmarks relative to wrists by a factor between [0.92, 1.08]
        to handle distance-from-camera and hand size differences without shifting arm pose.
        """
        tensor = tensor.clone()
        scale_factor = float(np.random.uniform(scale_range[0], scale_range[1]))
        hand_slices = [(100, 163), (163, 226)] if tensor.shape[-1] == 226 else [(1536, 1599), (1599, 1662)]
        T = tensor.shape[0]

        for start_idx, end_idx in hand_slices:
            hand_data = tensor[:, start_idx:end_idx].clone().view(T, 21, 3)
            # Detect frames where hand is present (wrist coordinates non-zero)
            detected_mask = (hand_data[:, 0, :] != 0.0).any(dim=-1)
            if not detected_mask.any():
                continue
            wrists = hand_data[:, 0:1, :]  # Wrist landmark pt 0
            scaled = wrists + scale_factor * (hand_data - wrists)
            hand_data[detected_mask] = scaled[detected_mask]
            tensor[:, start_idx:end_idx] = hand_data.view(T, -1)

        return tensor

    def _apply_hand_jittering(self, tensor: torch.Tensor, std: float = 0.006) -> torch.Tensor:
        """
        Adds subtle Gaussian noise specifically to hand coordinates while preserving missing detection zeros.
        """
        tensor = tensor.clone()
        hand_slice = tensor[:, 100:226] if tensor.shape[-1] == 226 else tensor[:, 1536:1662]
        noise = torch.randn_like(hand_slice) * std
        non_zero_mask = (hand_slice != 0.0).float()
        if tensor.shape[-1] == 226:
            tensor[:, 100:226] = hand_slice + (noise * non_zero_mask)
        else:
            tensor[:, 1536:1662] = hand_slice + (noise * non_zero_mask)
        return tensor

    def _apply_temporal_dropout(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Randomly masks out 1-2 entire frames (zero vectors) to simulate frame drop or sudden occlusion.
        """
        num_drop = int(np.random.randint(1, self.max_dropped_frames + 1))
        drop_indices = np.random.choice(tensor.shape[0], size=num_drop, replace=False)
        tensor = tensor.clone()
        tensor[drop_indices, :] = 0.0
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Dynamically load .npy file
        data = np.load(str(file_path)).astype(np.float32)

        # Trim high-dimensional features (1662 -> 226) if enabled
        if self.trim_features:
            data = trim_landmarks(data)

        # Ensure correct shape (T, D)
        if data.shape[0] != self.fixed_frames:
            if data.shape[0] > self.fixed_frames:
                indices = np.linspace(0, data.shape[0] - 1, self.fixed_frames, dtype=int)
                data = data[indices]
            else:
                pad = np.zeros((self.fixed_frames - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, pad])

        features = torch.from_numpy(data).float()

        # Richer On-the-Fly Dynamic Augmentation (applied during training mode)
        if self.is_train:
            # 1. Dynamic Temporal Warping (resampling between 24 to 36 frames)
            if np.random.rand() < 0.5:
                features = self._apply_dynamic_temporal_warping(features, min_frames=24, max_frames=36)
            # 2. Hand Scale Variation (scale relative to wrist in [0.92, 1.08])
            if np.random.rand() < 0.5:
                features = self._apply_hand_scale_variation(features, scale_range=(0.92, 1.08))
            # 3. Spatial Hand Jittering (subtle Gaussian noise to hand landmarks)
            if np.random.rand() < 0.5:
                features = self._apply_hand_jittering(features, std=0.006)
            # 4. Temporal Frame Dropout
            if np.random.rand() < 0.3:
                features = self._apply_temporal_dropout(features)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return features, label_tensor


def stratified_split(
    file_paths: List[Path],
    labels: List[int],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Path], List[int], List[Path], List[int], List[Path], List[int]]:
    """
    Performs a robust, reproducible stratified split across classes ensuring
    every gloss is balanced across train, validation, and test subsets.
    """
    rng = np.random.RandomState(seed)

    # Group sample paths by class label
    class_groups = {}
    for p, y in zip(file_paths, labels):
        class_groups.setdefault(y, []).append(p)

    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []

    for y, paths in sorted(class_groups.items()):
        paths = list(paths)
        rng.shuffle(paths)

        n_total = len(paths)
        n_val = int(round(n_total * val_split))
        n_test = int(round(n_total * test_split))

        # Ensure at least 1 sample in train if possible
        if n_val + n_test >= n_total:
            n_val = max(1, n_val - 1)
            n_test = max(1, n_test - 1)

        val_subset = paths[:n_val]
        test_subset = paths[n_val : n_val + n_test]
        train_subset = paths[n_val + n_test :]

        train_files.extend(train_subset)
        train_labels.extend([y] * len(train_subset))

        val_files.extend(val_subset)
        val_labels.extend([y] * len(val_subset))

        test_files.extend(test_subset)
        test_labels.extend([y] * len(test_subset))

    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def create_dataloaders(
    processed_dir: Union[str, Path] = "data/processed_data",
    label_map_path: Union[str, Path] = "data/label_map.json",
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    trim_features: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Discovers processed .npy files, performs stratified train/val/test splitting,
    creates PyTorch DataLoaders with training augmentations, and returns feature dimension.

    Returns:
        (train_loader, val_loader, test_loader, feature_dim)
    """
    processed_path = Path(processed_dir)
    label_map_file = Path(label_map_path)

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed directory '{processed_path}' does not exist.")
    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map file '{label_map_file}' does not exist.")

    with open(label_map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    all_file_paths = []
    all_labels = []

    # Traverse processed directory respecting label_map
    for class_name, class_idx in sorted(label_map.items()):
        class_folder = processed_path / class_name
        if not class_folder.exists():
            continue

        for npy_file in sorted(class_folder.glob("*.npy")):
            all_file_paths.append(npy_file)
            all_labels.append(class_idx)

    if not all_file_paths:
        raise RuntimeError(f"No .npy files found in '{processed_path}'.")

    # Peek at the first file to determine feature dimension
    sample_data = np.load(str(all_file_paths[0]))
    if trim_features:
        sample_data = trim_landmarks(sample_data)
    feature_dim = sample_data.shape[1] if sample_data.ndim == 2 else sample_data.shape[-1]

    # Perform Stratified Split
    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    ) = stratified_split(
        all_file_paths,
        all_labels,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    # Instantiate Datasets with feature trimming
    train_dataset = ISLDataset(train_paths, train_labels, is_train=True, trim_features=trim_features)
    val_dataset = ISLDataset(val_paths, val_labels, is_train=False, trim_features=trim_features)
    test_dataset = ISLDataset(test_paths, test_labels, is_train=False, trim_features=trim_features)

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, feature_dim


if __name__ == "__main__":
    print("=== Testing ISL Dataset and DataLoader Pipeline ===")

    processed_dir = "data/processed_data"
    label_map_path = "data/label_map.json"

    train_loader, val_loader, test_loader, feature_dim = create_dataloaders(
        processed_dir=processed_dir,
        label_map_path=label_map_path,
        batch_size=16,
        val_split=0.15,
        test_split=0.15,
        seed=42,
    )

    num_classes = len(set(train_loader.dataset.labels))
    print(f"Total Unique Classes:  {num_classes}")
    print(f"Train Dataset Samples: {len(train_loader.dataset)}")
    print(f"Val Dataset Samples:   {len(val_loader.dataset)}")
    print(f"Test Dataset Samples:  {len(test_loader.dataset)}")
    print(f"Feature Dimension:     {feature_dim}")
    print(f"Train Batches:         {len(train_loader)}")

    # 1. Fetch and verify a single batch from train_loader
    for batch_features, batch_labels in train_loader:
        print("\n--- Train Batch Verification ---")
        print(f"Batch Features Shape: {batch_features.shape}")
        print(f"Batch Features Dtype: {batch_features.dtype}")
        print(f"Batch Labels Shape:   {batch_labels.shape}")
        print(f"Batch Labels Dtype:   {batch_labels.dtype}")
        print(f"Sample Labels:        {batch_labels.tolist()[:8]}...")
        assert batch_features.shape == (batch_features.shape[0], 30, feature_dim), "Unexpected feature shape!"
        assert batch_features.dtype == torch.float32, "Features should be float32!"
        assert batch_labels.dtype == torch.int64, "Labels should be int64 (Long)!"
        assert not torch.isnan(batch_features).any(), "Found NaNs in batch features!"
        assert not torch.isinf(batch_features).any(), "Found Infs in batch features!"
        print("[Pass] Batch verification passed successfully!")
        break

    # 2. Verify individual augmentations on a single sample
    print("\n--- Testing Individual Augmentations on Raw Sample ---")
    sample_tensor, _ = train_loader.dataset[0]

    dataset_instance = train_loader.dataset

    # Dynamic Temporal Warping Test
    warped = dataset_instance._apply_dynamic_temporal_warping(sample_tensor, min_frames=24, max_frames=36)
    assert warped.shape == (30, feature_dim), "Dynamic temporal warping changed shape!"
    assert not torch.isnan(warped).any(), "NaN found in dynamic temporal warping!"
    print(f"[Pass] Dynamic temporal warping: shape {warped.shape} preserved.")

    # Hand Scale Variation Test
    scaled = dataset_instance._apply_hand_scale_variation(sample_tensor, scale_range=(0.92, 1.08))
    assert scaled.shape == (30, feature_dim), "Hand scale variation changed shape!"
    assert not torch.isnan(scaled).any(), "NaN found in hand scale variation!"
    print(f"[Pass] Hand scale variation: shape {scaled.shape} preserved.")

    # Spatial Hand Jittering Test
    jittered = dataset_instance._apply_hand_jittering(sample_tensor, std=0.006)
    assert jittered.shape == (30, feature_dim), "Hand jittering changed shape!"
    zeros_in_orig = (sample_tensor == 0.0)
    assert (jittered[zeros_in_orig] == 0.0).all(), "Jitter corrupted zero-padded landmarks!"
    print(f"[Pass] Hand jittering: shape {jittered.shape}, preserved {zeros_in_orig.sum().item()} zero entries.")

    # Temporal Dropout Test
    dropped = dataset_instance._apply_temporal_dropout(sample_tensor)
    assert dropped.shape == (30, feature_dim), "Dropout changed shape!"
    zero_frames = (dropped == 0.0).all(dim=1).sum().item()
    print(f"[Pass] Temporal dropout: shape {dropped.shape}, total all-zero frames: {zero_frames}.")

    print("\n=== All ISL Dataset Pipeline Tests Passed Successfully! ===")
