# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modified work Copyright 2025 Bowen Zheng
# The Chinese University of Hongkong, Shenzhen
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import sys
import math
import subprocess
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from pathlib import Path

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Simple augmentations (center crop / random crop) adapted from ADM

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0, rng=None):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    # Use an explicit RNG when provided (e.g., deterministic per-sample cropping).
    # Otherwise fall back to the global NumPy RNG (which can be controlled via seeding).
    rng = np.random if rng is None else rng
    smaller_dim_size = rng.randint(min_smaller_dim_size, max_smaller_dim_size + 1)

    # Repeated BOX downsampling preserves the original crop preprocessing.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = rng.randint(0, arr.shape[0] - image_size + 1)
    crop_x = rng.randint(0, arr.shape[1] - image_size + 1)
    return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_label  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
            ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_label = use_label
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_label else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._load_raw_image(raw_idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

    # ------------------------------------------------------------------------
    # Simple augmentations (center crop / random crop) as Dataset helpers,
    def _center_crop_arr(self, pil_image, image_size):
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
        """
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

    def _random_crop_arr(self, pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0, rng=None):
        min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
        max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
        rng = np.random if rng is None else rng
        smaller_dim_size = rng.randint(min_smaller_dim_size, max_smaller_dim_size + 1)

        # Repeated BOX downsampling preserves the original crop preprocessing.
        while min(*pil_image.size) >= 2 * smaller_dim_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=PIL.Image.BOX
            )

        scale = smaller_dim_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = rng.randint(0, arr.shape[0] - image_size + 1)
        crop_x = rng.randint(0, arr.shape[1] - image_size + 1)
        return PIL.Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        crop_type       = None, # None / 'center' / 'random'
        deterministic_crop = False, # If True and crop_type=='random', crop depends only on (crop_seed, raw_idx).
        crop_seed       = 0,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self._crop_type = crop_type
        self._resolution = resolution
        self._deterministic_crop = bool(deterministic_crop)
        self._crop_seed = int(crop_seed)

        # If labels are requested and no dataset.json is present, try to
        # create it in-place using dataset_tools.py (labels-only mode).
        use_label = bool(super_kwargs.get('use_label', False))
        if use_label:
            self._ensure_dataset_json()

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _ensure_dataset_json(self):
        """Ensure dataset.json with labels exists for this dataset path.

        If missing, call dataset_tools.py in labels-only mode to generate it.
        """
        # Directory: check for dataset.json at root.
        if os.path.isdir(self._path):
            json_path = os.path.join(self._path, 'dataset.json')
            if os.path.isfile(json_path):
                return
        # Zip: check if dataset.json is inside archive.
        elif self._file_ext(self._path) == '.zip':
            try:
                with zipfile.ZipFile(self._path, mode='r') as z:
                    if 'dataset.json' in z.namelist():
                        return
            except Exception:
                return
        else:
            return

        # Use the dataset preparation tool bundled with this package.
        project_root = Path(__file__).resolve().parent
        tools_path = project_root / 'scripts' / 'dataset_tools.py'
        if not tools_path.is_file():
            print(f"Warning: dataset_tools.py not found at {tools_path}, cannot auto-generate dataset.json labels.")
            return

        cmd = [
            sys.executable,
            str(tools_path),
            '--source', str(self._path),
            '--dest', str(self._path),
            '--labels-only',
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Warning: failed to generate dataset.json via dataset_tools.py: {e}")

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]

        # If no crop augmentation is requested, keep original fast path
        # (optionally using pyspng for PNGs).
        if self._crop_type is None:
            with self._open_file(fname) as f:
                if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                    image = pyspng.load(f.read())
                else:
                    image = np.array(PIL.Image.open(f))
        else:
            # Use PIL path so we can apply center/random crop.
            if self._resolution is None:
                raise ValueError("resolution must be set when using crop_type for ImageFolderDataset.")
            with self._open_file(fname) as f:
                pil_img = PIL.Image.open(f).convert("RGB")
                if self._crop_type == 'center':
                    pil_img = self._center_crop_arr(pil_img, self._resolution)
                elif self._crop_type == 'random':
                    rng = None
                    if self._deterministic_crop:
                        # Make cropping independent of call order / other RNG consumers.
                        # Note: this fixes the crop for a given raw_idx across epochs.
                        seed = (self._crop_seed + int(raw_idx)) % (1 << 31)
                        rng = np.random.RandomState(seed)
                    pil_img = self._random_crop_arr(pil_img, self._resolution, rng=rng)
                else:
                    raise ValueError(f"Unknown crop_type: {self._crop_type}")
                image = np.array(pil_img)

        # Normalize channel count to 3 (RGB) where possible.
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)  # HW => HWC (3-channel grayscale)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] > 3:
                image = image[:, :, :3]

        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
