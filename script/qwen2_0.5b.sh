#!/bin/bash
#SBATCH --job-name=qwen
#SBATCH --output=./stdout/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --qos=gpu-8

cd ../src
# 打印使用的程序的名字和路径
echo "Program name: src/qwen2.py"
echo "Starting at: $(date)"
python qwen2.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --save_dir ../results/Qwen2-0.5B-Instruct-zeroshot \
    --num_few_shot 0

python qwen2.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --save_dir ../results/Qwen2-0.5B-Instruct-fiveshot \
    --num_few_shot 5

