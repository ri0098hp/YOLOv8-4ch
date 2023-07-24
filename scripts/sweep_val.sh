#!/bin/sh

cd ~/workspace
conds=(day night hot inter cold)
models=(day night hot inter cold time season)
for cond in ${conds[@]}
do
  for model in ${models[@]}
  do
    yolo detect val data=data/All-Season-${cond}.yaml model=runs/detect/All-Season-${model}-4ch/weights/best.pt project=runs/val name=${cond}-by-${model};
  done;
done
