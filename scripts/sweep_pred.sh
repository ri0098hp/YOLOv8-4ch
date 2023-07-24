#!/bin/sh

cd ~/workspace
conds=(day-winter night summer)
models=(4 1 3)
for cond in ${conds[@]}
do
  for model in ${models[@]}
  do
      yolo predict source=datasets/demo_allseason/${cond} model=weights/All-Season-${model}ch-aug.pt project=runs/demo name=${cond}-${model}ch save=True
  done;
done
