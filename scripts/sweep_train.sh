#!/bin/bash

cd ~/workspace
data=All-Season.yaml

param=hsv_ir
for i in $(seq 0.0 0.1 1.0) # 0から0.1刻みで
do
  yolo data=${data} epochs=50 ${param}=${i} name=${param}@${i};
done
