#!/bin/bash

cd ~/workspace
conds=(hot inter cold season)
models=(hot inter cold season)
for cond in ${conds[@]}
do
  for model in ${models[@]}
  do
    yolo detect val data=All-Season-${cond}.yaml model=runs/season/All-Season-${model}-4ch/weights/best.pt name=${cond}-by-${model};
  done;
done
