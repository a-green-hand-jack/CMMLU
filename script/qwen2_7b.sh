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
# python qwen2.py \
#     --model_name_or_path Qwen/Qwen2-7B-Instruct \
#     --save_dir ../results_bio/Qwen2-7B-Instruct-zeroshot-finetuned-v2 \
#     --num_few_shot 0    \
#     --data_dir ../finetune_data_split   \
#     --cache_dir /root/autodl-fs/pre-trained-models/hub/ \
#     --peft_model_id /root/autodl-tmp/project/BIO/llama3-bio-edu/output/bio-train-6911/final_model

echo "Starting at: $(date)"
python qwen2.py \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --save_dir ../results_bio/Qwen2-7B-Instruct-fiveshot-finetuned-v2 \
    --num_few_shot 5    \
    --data_dir ../finetune_data_split   \
    --cache_dir /root/autodl-fs/pre-trained-models/hub/ \
    --peft_model_id /root/autodl-tmp/project/BIO/llama3-bio-edu/output/bio-train-6911/final_model
