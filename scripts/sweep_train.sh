#!/bin/sh

cd ~/workspace
data=data/All-Season.yaml

param=hsv_ir
for i in $(seq 0.0 0.1 1.0) # 0から0.1刻みで
do
  yolo cfg=cfg/yolov8.yaml data=${data} epochs=50 ${param}=${i} name=${param}@${i};
  sleep 30;
  rm -rf nohup.out
done
