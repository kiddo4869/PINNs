#!/bin/bash

cd ../

python train.py --seed 4869 --layers "[2,20,20,20,1]" --epochs 100000 --learning_rate 1e-3 --save_epoch_freq 5000
