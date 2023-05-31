"""
data/custom以下に設定したcfg.yamlを保存
ベースとなるcfgはDEFAULT_CFGで変更可能
"""

import glob
import os
from pathlib import Path
from pprint import pprint

from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import yaml_save

DEFAULT_CFG = "cfg/yolov8.yaml"
ROOT_DIR = "/home/suwako/workspace"


def main():
    os.chdir(ROOT_DIR)
    pprint(glob.glob("data/*.yaml"))
    data_name = input("name of data: ")

    cfg = f"cfg/custom/{data_name}.yaml"
    if not os.path.exists(cfg):
        cfg = DEFAULT_CFG
    args = get_cfg(cfg, None)

    args.data = f"data/{data_name}.yaml"

    data_dict = check_det_dataset(args.data)
    if "yaml_file" in data_dict:
        args.data = data_dict["yaml_file"]  # for validating 'yolo train data=url.zip' usage

    # Train dataloader ---------------------------------------------------------------------------------------
    print("pos_imgs_train:", args.get("pos_imgs_train"))
    print("neg_ratio_train:", args.get("neg_ratio_train"))

    while True:
        try:
            build_dataloader(cfg=args, img_path=data_dict, names=data_dict["names"], batch=64, mode="train")
        except PermissionError:
            pass

        os.makedirs(Path("cfg") / "custom", exist_ok=True)
        yaml_save(Path("cfg") / "custom" / f"{data_name}.yaml", vars(args))  # save run args

        # input number
        print("0: 決定, -1: 全データ, 自然数: ラベル有画像数")
        p = int(input("pos_imgs_train: "))
        if p == 0:
            break
        n = float(input("neg_ratio_train: "))

        # apply to dict
        if p != -1:
            args.pos_imgs_train = p
        elif args.get("pos_imgs_train"):
            del args.pos_imgs_train
        if n != -1:
            args.neg_ratio_train = n
        elif args.get("neg_ratio_train"):
            del args.neg_ratio_train

    # Validater dataloader ---------------------------------------------------------------------------------------
    print("pos_imgs_val:", args.get("pos_imgs_val"))
    print("neg_ratio_val:", args.get("neg_ratio_val"))

    while True:
        try:
            build_dataloader(cfg=args, img_path=data_dict, names=data_dict["names"], batch=64 * 2, rank=-1, mode="val")
        except PermissionError:
            pass

        yaml_save(Path("cfg") / "custom" / f"{data_name}.yaml", vars(args))  # save run args

        print("0: 決定, -1: 全データ, 0.1~1.0: ラベル無しデータの割合")
        p = int(input("pos_imgs_val: "))
        if p == 0:
            break
        n = float(input("neg_ratio_val: "))
        # apply to dict
        if p != -1:
            args.pos_imgs_val = p
        elif args.get("pos_imgs_val"):
            del args.pos_imgs_val
        if n != -1:
            args.neg_ratio_val = n
        elif args.get("neg_ratio_val"):
            del args.neg_ratio_val


if __name__ == "__main__":
    main()
