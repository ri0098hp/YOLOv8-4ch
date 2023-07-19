#!/bin/sh

cd ~/workspace
param=flipir
for i in $(seq 0.0 0.1 0.5) # 0から0.1刻みで1まで
do
  yolo cfg=cfg/yolov8-aug.yaml data=data/kaist-old.yaml epochs=20 project=runs/tune ${param}=${i} name=${param}@${i};
  sleep 5;
done


# conds=(day night hot inter cold time season)
# models=(4 1 3)
# for cond in ${conds[@]}
# do
#   for model in ${models[@]}
#   do
#     yolo cfg=cfg/custom/fujinolab-${cond}.yaml ch=${model} name=fujinolab-${cond}-${model}ch;
#   done;
# done
