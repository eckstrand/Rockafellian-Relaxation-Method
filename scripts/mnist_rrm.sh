#!/bin/bash

#SBATCH --job-name=rrm
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=dsag
#SBATCH --nodelist=compute-8-17
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/data/eric.eckstrand/out/rrm/%j.out

. /etc/profile
module load lang/miniconda3/4.10.3
module load app/cplex/12.8.0
source activate py3.9tf2.9

#python -m debugpy --wait-for-client --listen 0.0.0.0:54327 ./train.py \
python ./train.py \
--results_dir /data/eric.eckstrand/out/rrm/mnist_rrm_${SLURM_JOB_ID} \
--solver cplex \
--solver_exe /share/apps/cplex/12.8.0/cplex/bin/x86-64_linux/cplex \
--theta 0.5 \
--u_reg 'l1' \
--n_iterations 4 \
--epochs 2 \
--batch 100 \
--nn_opt 'sgd' \
--use_model 'fc_royset_norton' \
--swap_pct 0.1 \
--lr 0.1 \
--mu 0.5 \
--small_mnist \
--u_opt \
# --trn_val_tst \
# --adv_trn \
# --adv_tst \
# --eps_trn 1.0 \
# --eps_tst '0.0,0.2,0.4'\
