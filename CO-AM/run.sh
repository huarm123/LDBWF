#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore"

# resnet18, resnet50
export NET='bcnn'
export path='model'
export data='/home/raomei/dmy/fg-web-data/web-bird'
export N_CLASSES=200
export lr=0.01
export w_decay=1e-8
export m=0.4
#export a=0.001
export label_weight=0.6

python train.py --label_weight ${label_weight}  --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data} --lr ${lr} --w_decay ${w_decay} --epochs 60 --step 1 --cos --m ${m}

sleep 300

export lr=0.01
export w_decay=1e-5

python train.py --label_weight ${label_weight} --net ${NET} --n_classes ${N_CLASSES} --path ${path} --data_base ${data} --lr ${lr} --w_decay ${w_decay} --epochs 120 --step 2 --cos --m ${m}

