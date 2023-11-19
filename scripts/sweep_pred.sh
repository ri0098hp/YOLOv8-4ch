#!/bin/sh

cd ~/workspace
conds=(day night)
conds=(hot inter cold)
chs=(4 1 3)
for cond in ${conds[@]}
do
  for ch in ${chs[@]}
  do
      yolo predict source=datasets/demo_slides/${cond} model=runs/All-Season/channel/All-Season-${cond}-${ch}ch/weights/best.pt project=runs/demo name=${cond}-${ch}ch save=True
  done;
done
