#!/usr/bin/env bash
for model_type in 'resnet18' 'alexnet' 'regression' 'mlp'
do
  python ../main.py --train --do_test --use_gpu --n_epochs 30 --model_type $model_type --dataset_root ../data
done