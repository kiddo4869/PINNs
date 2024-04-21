#!/bin/bash

cd ../

python train.py --seed 0 --layers "[2,20,20,20,1]" --epochs 40000 --learning_rate 1e-3 \
                --save_epoch_freq 1000 --log_epoch_freq 1000 \
                --pinn --output_path ./outputs/pinn_no_log
