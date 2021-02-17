#!/bin/bash
# Job name:
#SBATCH --job-name=1024benchmark
#
# Account:
#SBATCH -A m3691
#
# Partition:
#SBATCH -C gpu
#
# Number of nodes:
#SBATCH -N 1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH -c 80
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH -G 8
#
# Wall clock limit:
#SBATCH --time=4:00:00
#    --checkpoint last \
#   --init_batch_size 1024 \
## Command(s) to run (example):
EXP_NAME=wolf_4level1024
rsync -h --info=progress2 /global/cscratch1/sd/bobotran/medical_wolf/covidxct_v1/covidxct_v1.tar.gz /tmp/hsperfdata_bobotran/
tar -xzf /tmp/hsperfdata_bobotran/covidxct_v1.tar.gz -C /tmp/hsperfdata_bobotran/
conda activate cdcgen
python -u main.py \
    --checkpoint last \
    --lr 1e-4 \
    --beta1 0.5 \
    --lr_decay 0.99 \
    --init_batch_size 1024 \
    --dataset ct \
    --name $EXP_NAME \
    --gpus 8 \
    --batch_steps 2 \
    --config configs/4level.json \
>& out/${EXP_NAME}.out
