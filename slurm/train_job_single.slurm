#!/bin/bash
#SBATCH --job-name=train_nanoVLM
#SBATCH --output=logs/train_nanoVLM/%A_%a.out
#SBATCH --error=logs/train_nanoVLM/%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH -q mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=k.schwethelm@tum.de

cd /dss/dsshome1/0D/go68tos2/looped_nanoVLM
source .venv/bin/activate

# Set cache directories (datasets as subdirectory of HF_HOME)
export HF_HOME="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000/LLMs/.cache/huggingface"
export HF_DATASETS_CACHE="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000/LLMs/.cache/huggingface/datasets"

torchrun --nproc_per_node=1 train.py \
    --vlm_checkpoint_path /dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000/kristian/looped_nanoVLM/checkpoints \
    --log_wandb True \
    --wandb_entity "medtum" \
    --batch_size 2 \
    --gradient_accumulation_steps 8