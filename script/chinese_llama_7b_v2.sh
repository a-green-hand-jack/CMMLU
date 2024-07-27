#!/bin/bash
#SBATCH --job-name=chinese_llama_7b_eval
#SBATCH --output=./stdout/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --qos=gpu-8

cd ../src

for i in {0..5}; do
# 设置网络环境
export HF_ENDPOINT=https://hf-mirror.com
# 设置模型缓存路径
export HF_HOME=/root/autodl-fs/pre-trained-models/
# 这个方法不知道为什么没有起效果
python chinese_llama_alpaca.py \
    --model_name_or_path baicai003/Llama3-Chinese_v2 \
    --save_dir ../results/Llama3-Chinese_v2 \
    --num_few_shot $i
done
