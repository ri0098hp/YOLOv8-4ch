"""
dataのパス含めて全て絶対パスでないと動かないっぽい...?
"""

from ray import tune

from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
result = model.tune(
    data=str("/home/suwako/workspace/data/all-tune.yaml"),
    space={
        "hsv_ir": tune.uniform(0.0, 0.9),
    },
    train_args={
        "batch": 64,
        "epochs": 50,
        "cfg": "/home/suwako/workspace/cfg/yolov8.yaml",
        "pos_imgs_train": 3000,
        "neg_ratio_train": 0.5,
        "pos_imgs_val": 3000,
        "neg_ratio_val": 0.5,
    },
    gpu_per_trial=1,
)
