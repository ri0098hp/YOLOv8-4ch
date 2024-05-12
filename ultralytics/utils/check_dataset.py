"""
ultralytics/cfg/custom以下に設定したcfg.yamlを保存
"""

from pathlib import Path


from ultralytics.cfg import cfg2dict, get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

DEFAULT_CFG = "ultralytics/cfg/default.yaml"
ROOT_DIR = "/home/suwako/workspace"


def check_dataset():
    fps_cfg = sorted(Path("ultralytics/cfg/custom/").glob("*.yaml"))
    for i, fp_cfg in enumerate(fps_cfg):
        print(i + 1, "\t", fp_cfg.name)
    id = input("index of cfg [0: default cfg]: ")
    fp_cfg = fps_cfg[int(id) - 1] if id != 0 else DEFAULT_CFG
    print(fp_cfg, "\n")

    args = {**cfg2dict(DEFAULT_CFG), **cfg2dict(fp_cfg)}
    args = get_cfg(args, None)

    if id == 0:
        fps_data = sorted(Path("ultralytics/cfg/datasets/").glob("*.yaml"))
        for i, fp_data in enumerate(fps_data):
            print(i, "\t", fp_data.name)
        args.data = fps_data[int(input("index of data: "))]

    data_dict = check_det_dataset(args.data)
    if "yaml_file" in data_dict:
        args.data = data_dict["yaml_file"]  # for validating 'yolo train data=url.zip' usage

    # Train dataloader ---------------------------------------------------------------------------------------
    print("pos_imgs_train:", args.get("pos_imgs_train"))
    print("neg_ratio_train:", args.get("neg_ratio_train"))
    while True:
        try:
            build_yolo_dataset(args, data_dict, 1, data_dict, mode="train")
        except PermissionError:
            pass

        # input number
        print("0: exit, -1: all data")
        p = int(input("num for pos_imgs_train (int) : "))
        if p == 0:
            break
        n = float(input("neg_ratio_train (float) : "))

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
    try:
        build_yolo_dataset(args, data_dict, 1, data_dict, mode="val")
    except PermissionError:
        pass

    print(f"\nplease add to {fp_cfg}")
    try:
        if args.pos_imgs_train is not None:
            print("pos_imgs_train:", args.pos_imgs_train)
    except Exception:
        pass
    try:
        if args.neg_ratio_train is not None:
            print("neg_ratio_train:", args.neg_ratio_train)
    except Exception:
        pass


if __name__ == "__main__":
    check_dataset()
