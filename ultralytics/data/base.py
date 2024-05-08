# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

from .utils import HELP_URL, IMG_FORMATS


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.data_path = Path(img_path["data_path"])
        os.makedirs(self.data_path / "cache", exist_ok=True)

        self.rgb_folder = img_path.get("rgb_folder")
        self.fir_folder = img_path.get("fir_folder")
        self.labels_folder = img_path["labels_folder"]
        self.ch = img_path["ch"]
        self.is_train = "train" if "train" in prefix else "val"
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files()
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class

        # start of custom code --------------------------------------------------------------------------------------
        # limitting numbers of data on training
        if self.is_train == "train":
            # limitting numberrs of data on testing
            # positive imgs
            pos_id = [i for i, label in enumerate(self.labels) if label["bboxes"].any()]  # number of found labels
            pos_num = len(pos_id)  # number of found labels
            if hyp.get("pos_imgs_train") is not None:
                target_num = hyp.get("pos_imgs_train")
                assert (
                    target_num <= pos_num
                ), f"{prefix}please check your hyp[pos_imgs_train], must be less than {pos_num}"
                random.seed(hyp.get("seed") + 2 + LOCAL_RANK)
                # 現在の有効ラベル群から消去したいラベル, "現在のラベル数-指定のラベル数" 個分をポインタで指定
                idx = random.sample(pos_id, pos_num - target_num)
                for i in sorted(idx, reverse=True):
                    self.labels.pop(i), self.im_files.pop(i)
                pos_num = target_num

            # negative imgs
            neg_id = [i for i, label in enumerate(self.labels) if not label["bboxes"].any()]  # missed labels
            neg_num = len(neg_id)  # number of missed labels
            emsg = f"neg_ratio_train must be less than {neg_num / (pos_num + neg_num)}"
            print(prefix + emsg)
            if hyp.get("neg_ratio_train"):
                r = hyp.get("neg_ratio_train")
                assert 1 - r > 0, emsg
                target_num = int(pos_num * (r / (1 - r)))
                assert target_num <= neg_num, emsg
                random.seed(hyp.get("seed") + 3 + LOCAL_RANK)
                # 現在の有効ラベル群から消去したいラベル, "現在のラベル数-有効ラベル数*指定比率" 個分をポインタで指定
                idx = random.sample(neg_id, int(neg_num - target_num))
                for i in sorted(idx, reverse=True):
                    self.labels.pop(i), self.im_files.pop(i)
        else:
            # limitting numberrs of data on testing
            # positive imgs
            pos_id = [i for i, label in enumerate(self.labels) if label["bboxes"].any()]  # number of found labels
            pos_num = len(pos_id)  # number of found labels
            if hyp.get("pos_imgs_val"):
                target_num = hyp.get("pos_imgs_val")
                assert (
                    target_num <= pos_num
                ), f"{prefix}please check your hyp[pos_imgs_val], must be less than {pos_num}"
                random.seed(hyp.get("seed") + 4 + LOCAL_RANK)
                # 現在の有効ラベル群から消去したいラベル, "現在のラベル数-指定のラベル数" 個分をポインタで指定
                idx = random.sample(pos_id, pos_num - target_num)
                for i in sorted(idx, reverse=True):
                    self.labels.pop(i), self.im_files.pop(i)
                pos_num = target_num

            # negative imgs
            neg_id = [i for i, label in enumerate(self.labels) if not label["bboxes"].any()]  # missed labels
            neg_num = len(neg_id)  # number of missed labels
            emsg = f"neg_ratio_val must be less than {neg_num / (pos_num + neg_num)}"
            print(prefix + emsg)
            if hyp.get("neg_ratio_val"):
                r = hyp.get("neg_ratio_val")
                assert 1 - r > 0, emsg
                target_num = int(pos_num * (r / (1 - r)))
                assert target_num <= neg_num, emsg
                random.seed(hyp.get("seed") + 5 + LOCAL_RANK)
                # 現在の有効ラベル群から消去したいラベル, "現在のラベル数-有効ラベル数*指定比率" 個分をポインタで指定
                idx = random.sample(neg_id, int(neg_num - target_num))
                for i in sorted(idx, reverse=True):
                    self.labels.pop(i), self.im_files.pop(i)

        random.seed(hyp.get("seed") + 1 + LOCAL_RANK)
        # end of custom code --------------------------------------------------------------------------------------

        self.ni = len(self.labels)

        # rect stuff
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # RGB, FIRの整理とデータ数の保持
        if self.ch == 1:
            nRGB = 0
            nFIR = len(self.im_files)
        elif self.ch == 3:
            nRGB = len(self.im_files)
            nFIR = 0
        elif self.ch == 2 or self.ch == 4:  # loading FIR images from RGB image path
            src = os.sep + self.rgb_folder + os.sep
            dst = os.sep + self.fir_folder + os.sep
            self.im_files_ir = [x.replace(src, dst) for x in self.im_files]
            nRGB = len(self.im_files)
            nFIR = len(self.im_files_ir)

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images
        if cache == "ram" and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

        # save log files of loading
        log_path = self.data_path / "cache" / f"{self.is_train}_log.txt"
        msg = (
            f"{prefix}##########################\n"
            f"{prefix}{self.is_train} data has ...\n"
            f"{prefix}RGB: {nRGB} files\n"
            f"{prefix}FIR: {nFIR} files\n"
            f"{prefix}instance: {sum(len(data.get('bboxes')) for data in self.labels)} targets\n"
            f"{prefix}##########################"
        )
        print(msg)
        with open(log_path, "w") as f:
            f.write(msg.replace(prefix, ""))
            print(f"{prefix}DataLoader info save on: {log_path}")

    def get_img_files(self):
        """Read image files."""
        try:
            # loading images
            base = self.fir_folder if self.ch == 1 else self.rgb_folder
            if "train" in self.is_train:
                target = self.data_path / "train" / "**" / base / "*.*"
            else:
                target = self.data_path / "val" / "**" / base / "*.*"
            im_files = sorted(glob.iglob(str(target), recursive=True))
            im_files = [x for x in im_files if x.split(".")[-1].lower() in IMG_FORMATS]
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {self.data_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            # im_files = im_files[: round(len(im_files) * self.fraction)]
            num_elements_to_select = round(len(im_files) * self.fraction)
            im_files = random.sample(im_files, num_elements_to_select)
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                if self.ch == 1:
                    f = self.im_files[i]
                    im = cv2.imread(f, 0)  # gray
                elif self.ch == 4:
                    f = self.im_files[i]
                    im_rgb = cv2.imread(f)  # BGR
                    f = self.im_files_ir[i]
                    im_fir = cv2.imread(f, 0)  # gray
                    # im_fir = hist_eq(im_fir) # histogram equalization
                    im = cv2.merge((im_rgb, im_fir))  # combine rgb + ir
                else:
                    f = self.im_files[i]
                    im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {cache})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, "
                f"{'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Example:
            ```python
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
            ```
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Note:
            Ensure output is a dictionary with the following keys:
            ```python
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
            ```
        """
        raise NotImplementedError
