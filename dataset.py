import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Union

import imageio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.autonotebook import tqdm


def _add_channels(img: np.ndarray, total_channels: int = 3) -> np.ndarray:
    for _ in range(3 - len(img.shape)):
        img = np.expand_dims(img, axis=-1)
    for _ in range(total_channels - img.shape[-1]):
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


class TinyImageNetPaths:
    def __init__(self, root_dir: Union[str, Path]):
        train_path = os.path.join(root_dir, "train")
        val_path = os.path.join(root_dir, "val")
        test_path = os.path.join(root_dir, "test", "images")

        wnids_path = os.path.join(root_dir, "wnids.txt")
        words_path = os.path.join(root_dir, "words.txt")

        self._make_paths(train_path, val_path, test_path, wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path, wnids_path, words_path):
        self.ids = []
        with open(wnids_path, "r") as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, "r") as wf:
            for line in wf:
                nid, labels = line.split("\t")
                labels = list(map(lambda x: x.strip(), labels.split(",")))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            "train": [],  # [img_path, id, nid, box]
            "val": [],  # [img_path, id, nid, box]
            "test": [],  # img_path
        }

        # Get the test paths
        self.paths["test"] = list(
            map(lambda x: os.path.join(test_path, x), os.listdir(test_path))
        )
        # Get the validation paths and labels
        with open(os.path.join(val_path, "val_annotations.txt")) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, "images", fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths["val"].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + "_boxes.txt")
            imgs_path = os.path.join(train_path, nid, "images")
            label_id = self.ids.index(nid)
            with open(anno_path, "r") as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths["train"].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        mode: str = "train",
        preload: bool = True,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        IMAGE_SHAPE: Tuple = (64, 64, 3),
    ):
        tinp = TinyImageNetPaths(root_dir)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = IMAGE_SHAPE

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[: self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros(
                (self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32
            )
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                if mode != "test":
                    img = imageio.imread(s[0])
                else:
                    img = imageio.imread(s)
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != "test":
                    self.label_data[idx] = s[self.label_idx]

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = None if self.mode == "test" else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            img = _add_channels(img)
            lbl = None if self.mode == "test" else s[self.label_idx]
        if self.transform:
            img = img.astype("uint8")
            img = self.transform(img)
        return (img, lbl)
