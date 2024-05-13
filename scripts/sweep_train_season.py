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


subsets = ["hot", "inter", "cold", "season"]
chs = ["4ch", "3ch", "1ch", "2stream"]

for ch in chs:
    for subset in subsets:
        param = {
            "cfg": f"{subset}.yaml",
            "data": "All-Season.yaml" if subset == "season" else f"All-Season-{subset}.yaml",
            "ch": 4 if "2st" in ch else int(ch[0]),
            "name": f"All-Season-{subset}-{ch}",
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
