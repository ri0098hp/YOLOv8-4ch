import gc

import torch

from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# import cleaml if exists
try:
    assert SETTINGS["clearml"] is True  # verify integration is enabled
    import clearml
    from clearml import Task

    assert hasattr(clearml, "__version__")  # verify package is not directory

except (ImportError, AssertionError):
    clearml = None


subsets = ["hot", "inter", "cold", "mix"]
chs = ["4ch", "2stream"]

for ch in chs:
    for subset in subsets:
        param = {
            "data": "All-Season-tiny.yaml" if subset == "mix" else f"All-Season-tiny-{subset}.yaml",
            "ch": 4 if "2st" in ch else int(ch[0]),
            "name": f"All-Season-{subset}-{ch}",
            "pos_imgs_train": 960,
            "neg_ratio_train": 0.2,
            "project": "runs/sweep_season",
        }

        # Load a model
        model = "yolov8s.yaml" if "ch" in ch else f"yolov8s-{ch}.yaml"
        model = YOLO(model)

        # Use the model
        model.train(**param)

        # Release GPU memory and clear cache
        del model
        torch.cuda.empty_cache()
        gc.collect()

        if clearml and Task.current_task():
            Task.current_task().close()
