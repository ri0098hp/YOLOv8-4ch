import gc
from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path("runs/sweep_season")

metrics = [["Model", "AP@Hot", "AP@Inter", "AP@Cold"]]
model_subsets = ["hot", "inter", "cold", "mix"]
test_subsets = ["hot", "inter", "cold"]
suffs = ["4ch", "2stream"]  # 4ch or 2stream

for suffix in suffs:
    for model_subset in model_subsets:
        row = [model_subset]
        for test_subset in test_subsets:
            param = {
                "data": f"All-Season-tiny-{test_subset}.yaml",
                "batch": 1,
                "half": True,
                "plots": True if model_subset == "mix" else False,
                "name": f"{model_subset}_{test_subset}",
            }

            # Load a model
            model = YOLO(ROOT / f"All-Season-{model_subset}-{suffix}" / "weights" / "best.pt")

            # Use the model
            metric = model.val(**param)
            row.append(round(metric.results_dict["metrics/mAP50-95(B)"] * 1e2, 1))

            # Release GPU memory and clear cache
            del model
            torch.cuda.empty_cache()
            gc.collect()
        metrics.append(row)

    import pandas as pd

    df = pd.DataFrame(metrics)
    df.to_csv(f"runs/metrics_{suffix}.csv")
