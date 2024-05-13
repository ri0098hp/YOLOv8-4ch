import gc
from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path("runs/detect")

metrics = [["Model", "AP@Hot", "AP@Inter", "AP@Cold"]]
model_subsets = ["hot", "inter", "cold", "season"]
test_subsets = ["hot", "inter", "cold"]
suffix = "2stream"

for model_subset in model_subsets:
    row = [model_subset]
    for test_subset in test_subsets:
        param = {
            "data": f"All-Season-{test_subset}.yaml",
            "batch": 1,
            "plots": False,
        }

        # Load a model
        model = YOLO(ROOT / f"All-Season-{model_subset}-{suffix}" / "weights" / "best.pt")

        # Use the model
        metric = model.val(**param)
        row.append(round(metric.results_dict["metrics/mAP50(B)"] * 1e2, 1))

        # Release GPU memory and clear cache
        del model
        torch.cuda.empty_cache()
        gc.collect()
    metrics.append(row)

import pandas as pd

df = pd.DataFrame(metrics)
df.to_csv(f"metrics_{suffix}.csv")
