# Ultralytics YOLO 🚀, GPL-3.0 license

import glob
import math
import os
import random
import re
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import NUM_THREADS, TQDM_BAR_FORMAT
from .utils import HELP_URL, IMG_FORMATS, LOCAL_RANK, RANK


class BaseDataset(Dataset):
    """Base Dataset.
    Args:
        img_path (str): image path.
        pipeline (dict): a dict of image transforms.
        label_path (str): label path, this can also be an ann_file or other custom label path.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=None,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.5,
        single_cls=False,
    ):
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
        self.im_files = self.get_img_files()
        self.labels = self.get_labels()
        if self.single_cls:
            self.update_labels(include_class=[])

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
                random.seed(hyp.get("seed") + 2 + RANK)
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
                random.seed(hyp.get("seed") + 3 + RANK)
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
                random.seed(hyp.get("seed") + 4 + RANK)
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
                random.seed(hyp.get("seed") + 5 + RANK)
                # 現在の有効ラベル群から消去したいラベル, "現在のラベル数-有効ラベル数*指定比率" 個分をポインタで指定
                idx = random.sample(neg_id, int(neg_num - target_num))
                for i in sorted(idx, reverse=True):
                    self.labels.pop(i), self.im_files.pop(i)

        random.seed(hyp.get("seed") + 1 + RANK)
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

        # cache stuff
        self.ims = [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # transforms
        self.transforms = self.build_transforms(hyp=hyp)

        # save log files of loading
        log_path = self.data_path / "cache" / f"{self.is_train}_log.txt"
        msg = (
            "##########################\n"
            f"{self.is_train} data has ...\n"
            f"RGB: {nRGB} files\n"
            f"FIR: {nFIR} files\n"
            f"instance: {sum(len(data.get('bboxes')) for data in self.labels)} targets\n"
            "##########################"
        )
        print(msg)
        with open(log_path, "w") as f:
            f.write(msg)
            print(f"{prefix}DataLoader info save on: {log_path}")

    def get_img_files(self):
        """Read image files."""
        try:
            # loading RGB images
            if "kaist" in str(self.data_path):  # kaistの場合はtrainとvalフォルダで切り分け (先行研究)
                base = self.fir_folder if self.ch == 1 else self.rgb_folder
                if "train" in self.is_train:
                    target = self.data_path / "train" / "**" / base / "*.*"
                else:
                    target = self.data_path / "val" / "**" / base / "*.*"
                im_files = sorted(glob.iglob(str(target), recursive=True))
                im_files = [x for x in im_files if x.split(".")[-1].lower() in IMG_FORMATS]
            else:  # 自作データセットの場合はディレクトリごとに割合で振り分け
                target = self.data_path / "**" / self.rgb_folder
                dirs = sorted(glob.iglob(f"{target}/", recursive=True))
                dirs = [x.replace(self.rgb_folder + os.sep, "") for x in dirs]
                if self.ch == 1:  # FIR only
                    im_files = self.data_distributor(dirs, self.fir_folder)
                else:  # RGB only, RGB-FIR
                    im_files = self.data_distributor(dirs, self.rgb_folder)
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {self.data_path}\n{HELP_URL}") from e
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)"""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = segments[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, resized hw)
        im, fn = self.ims[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                if self.ch == 1:
                    f = self.im_files[i]
                    im = cv2.imread(f, 0)  # gray
                elif self.ch == 4:
                    f = self.im_files[i]
                    im_rgb = cv2.imread(f)  # BGR
                    f = self.im_files_ir[i]
                    im_fir = cv2.imread(f, 0)  # gray
                    im = cv2.merge((im_rgb, im_fir))  # combine rgb + ir
                else:
                    f = self.im_files[i]
                    im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images(self, cache):
        # cache images to memory or disk
        gb = 0  # Gigabytes of cached images
        self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = tqdm(enumerate(results), total=self.ni, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == "disk":
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({gb / 1E9:.1f}GB {cache})"
            pbar.close()

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def set_rectangle(self):
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
        return self.transforms(self.get_label_info(index))

    def get_label_info(self, index):
        label = self.labels[index].copy()
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label = self.update_labels_info(label)
        return label

    def __len__(self):
        return len(self.labels)

    def update_labels_info(self, label):
        """custom your label format here"""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # training transforms
                return Compose([])
            else:
                # val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
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
        """
        raise NotImplementedError

    # ADD: by okuda ------------------------------------------------------------------------------------------------
    def split_list(self, list: list, n: int) -> list:
        """
        配列を均等にn分割する
        """
        list_size = len(list)
        a = list_size // n
        b = list_size % n
        return [list[i * a + (i if i < b else b) : (i + 1) * a + (i + 1 if i < b else b)] for i in range(n)]

    def show_selected(self, dir: str, idx: list):
        """
        振り分けを可視化する関数
        """
        msg = f"{dir}: "
        for i in range(10):
            if i in idx:
                msg += "■"
            else:
                msg += "□"
        print(msg)

    def data_distributor(self, dirs, trg_folder):
        """
        学習画像とテスト画像を振り分け
        """
        fs = []  # ファイルパス
        if "train" in self.prefix:
            print("□ is train group, ■ is val group")
        for dir in dirs:  # 日付ディレクトリごとに探索
            # RGBフォルダ下の画像を探索
            f = sorted(glob.iglob(os.path.join(dir, trg_folder, "*.*"), recursive=True))
            f = [x for x in f if x.split(".")[-1].lower() in IMG_FORMATS]
            f.sort(key=lambda s: int(re.search(r"(\d+)\.", s).groups()[0]))  # 自然数で並び替え

            # train と test の振り分け - 再現性のためフォルダからハッシュ値を計算しシフト
            spl = self.split_list(f, 10)
            idx_train = [0, 1, 2, 4, 6, 7, 8]
            idx_val = [3, 5, 9]
            # idx_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # test all images
            try:
                d = int(re.sub(r"\D", "", dir))
            except Exception:
                d = ord(dir[-2])
            idx_train = list(map(lambda x: (x + d) % 10, idx_train))
            idx_val = list(map(lambda x: (x + d) % 10, idx_val))
            if "train" in self.prefix:
                self.show_selected(dir, idx_val)
                for id in idx_train:
                    fs += spl[id]
            else:
                for id in idx_val:
                    fs += spl[id]
        return fs
