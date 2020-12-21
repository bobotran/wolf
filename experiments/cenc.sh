#!/bin/bash
# Job name:
#SBATCH --job-name=adms
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
## Command(s) to run (example):
#conda activate cdcgen
#
#    --recover 0 \
EXP_NAME=svhn10_adms
python -u train.py \
    --config configs/mnist/glow/glow-extra.json \
    --lr_decay 0.99997 \
    --epochs 15000 --valid_epochs 10 \
    --batch_size 256 --batch_steps 4 --eval_batch_size 1000 --init_batch_size 2048 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 50 --weight_decay 1e-6 --grad_clip 200000 \
    --image_size 32 --n_bits 8 \
    --data_path svhn --model_path ${EXP_NAME} \
    --dataset svhn \
&> ${EXP_NAME}.out
