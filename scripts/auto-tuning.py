"""
dataのパス含めて全て絶対パスでないと動かないっぽい...?
"""

import pprint

import numpy as np
from ultralytics import YOLO

results = []
for arg in np.arange(0.0, 1.0, 0.1):
    print(arg)
    model = YOLO("yolov8n.yaml")
    model.train(cfg="cfg/yolov8-aug.yaml", data="data/kaist-old.yaml", epochs=20, hsv_ir=float(arg))
    results.append(model.val())

pprint.pprint(results)
print(type(results[0]))
