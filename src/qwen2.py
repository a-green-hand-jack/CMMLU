import os
import torch
import numpy as np
import argparse
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# 设置HF_HOME环境变量
os.environ['HF_HOME'] = "/root/autodl-fs/pre-trained-models/"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
device = "cuda" # the device to load the model onto
def is_eval_success(args) -> bool:
    """
    Check if the evaluation is successful.

    This function checks whether the evaluation of all subjects is successfully completed by verifying the existence of result files in the specified directory.
    
    Parameters:
    args: An argument object containing the directory path and other information.
    
    Returns:
    bool: Returns True if the evaluation results of all subjects are present, otherwise returns False.
    """
    """judege if eval task is success by checking the result dir"""
    # Sort the list of subject names by reading the CSV files in the test directory
    subjects = sorted(
        [f.split(".csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test/"))]
    )
    
    # Construct the path of the directory where the results are saved
    abs_save_dir = f"{args.save_dir}_{args.num_few_shot}_shot"
    
    # If the directory does not exist, the evaluation is not considered successful
    if not os.path.exists(abs_save_dir):
        return False
    
    # Iterate through all subjects
    for subject in subjects:
        # Construct the path of the result file for the subject
        out_file = os.path.join(abs_save_dir, f"results_{subject}.csv")
        
        # If the result file for any subject does not exist, the evaluation is not considered successful
        if not os.path.exists(out_file):
            # If any result file NOT exist, the eval isn't finished
            return False
    
    # If the result file for all subjects exists, the evaluation is considered successful
    return True


def init_model(args):
    """
    Initialize the language model.
    
    This function initializes a causal language model (CLM) based on the pre-trained model name or path provided by the arguments.
    It sets the model to use half-precision floating-point (float16) for computation to improve efficiency and reduces memory usage.
    
    Parameters:
    - args: An argument namespace containing the model name or path and other related configuration information.
    
    Returns:
    - model: The initialized language model.
    """
    """Initialize models"""
    # Load the pre-trained model using the provided model name or path, and configure it to trust remote code.
    # Automatically determine the device mapping for model deployment, and set the tensor data type to half-precision floating-point (float16).
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Import the generation configuration of the pre-trained model to ensure the model's generation behavior is consistent with the pre-trained model.
    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    
    return model


def eval(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot):
    """
    评估模型在特定主题上的表现。

    使用few-shot学习方法，根据开发集生成提示(prompt)，然后在测试集上对模型进行评估。

    参数:
    model: 训练好的模型，用于预测。
    tokenizer: 用于将文本转换为模型输入的tokenizer。
    subject: 评估的主题。
    dev_df: 开发集DataFrame。
    test_df: 测试集DataFrame。
    num_few_shot: few-shot学习中使用的样本数量。
    max_length: 输入序列的最大长度。
    cot: 是否包含选项在提示中。

    返回:
    准确率、所有预测结果的列表和一个占位符None。
    """
    # 将选项文本转换为tokenizer编码
    choice_ids = [tokenizer(choice)["input_ids"][0] for choice in choices]
    cors = []  # 存储预测结果与真实标签是否一致
    all_conf = []  # 存储所有预测的置信度
    all_preds = []  # 存储所有预测的结果
    answers = choices[: test_df.shape[1] - 2]  # 获取答案列表

    # 遍历测试集中的每个样本
    for i in range(test_df.shape[0]):
        # 根据当前样本格式化示例，生成prompt的结尾部分
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        # 根据开发集和当前样本生成完整的prompt
        prompt = gen_prompt(
            dev_df=dev_df,
            subject=subject,
            prompt_end=prompt_end,
            num_few_shot=num_few_shot,
            tokenizer=tokenizer,
            max_length=max_length,
            cot=cot,
        )
        label = test_df.iloc[i, test_df.shape[1] - 1]  # 获取当前样本的真实标签

        # 关闭梯度计算，提高推理效率
        with torch.no_grad():
            # 将prompt转换为模型输入的格式
            input_ids = tokenizer([prompt], padding=False)["input_ids"]
            input_ids = torch.tensor(input_ids, device=model.device)
            # 获取模型预测的logits
            logits = model(input_ids)["logits"]
            last_token_logits = logits[:, -1, :]  # 获取最后一个token的logits
            # 如果logits的类型是低精度浮点数，则转换为float32以提高计算精度
            if last_token_logits.dtype in {torch.bfloat16, torch.float16}:
                last_token_logits = last_token_logits.to(dtype=torch.float32)
            # 根据选项的ids提取对应logits
            choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
            # 计算模型对真实标签的置信度
            conf = softmax(choice_logits[0])[choices.index(label)]
            # 预测结果为logits最大值对应的选项
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_logits[0])]
        
        # 更新预测结果、置信度和预测正确与否的列表
        all_preds += pred
        all_conf.append(conf)
        cors.append(pred == label)

    # 计算预测准确率
    acc = np.mean(cors)
    # 打印准确率结果
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    # 返回准确率、所有预测结果和一个占位符None
    return acc, all_preds, None


def eval_chat(
    model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot
):
    """
    评估模型在聊天任务上的性能。
    
    参数:
    model: 预训练的语言模型。
    tokenizer: 用于编码和解码的分词器。
    subject: 聊天的主题。
    dev_df: 开发集数据框，用于生成提示。
    test_df: 测试集数据框，包含用于评估的问答对。
    num_few_shot: 少样本学习中使用的样本数量。
    max_length: 生成答案的最大长度。
    cot: 是否包含上下文提示。
    
    返回:
    acc: 模型预测的准确率。
    all_preds: 所有预测的答案。
    None: 本函数不返回第三个值，用于保持函数签名与预期一致。
    """
    """
    eval Qwen/Qwen1.5-7B-Chat
    ref: https://github.com/QwenLM/Qwen1.5?tab=readme-ov-file#quickstart
    """
    # 初始化准确率列表和所有预测答案的列表
    cors = []
    all_preds = []
    # 从测试集中提取答案，用于后续比较
    answers = choices[: test_df.shape[1] - 2]

    # 遍历测试集中的每个样本
    for i in tqdm(range(test_df.shape[0])):
        # 根据当前样本格式化示例，不包括答案
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        # 生成用于模型提示的完整prompt
        prompt = gen_prompt(
            dev_df=dev_df,
            subject=subject,
            prompt_end=prompt_end,
            num_few_shot=num_few_shot,
            tokenizer=tokenizer,
            max_length=max_length,
            cot=cot,
        )
        # 获取当前样本的正确答案
        label = test_df.iloc[i, test_df.shape[1] - 1]

        # 使用tokenizer处理生成的prompt，准备输入模型
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # 将处理后的文本转换为模型输入格式
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 通过模型生成答案
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        # 提取生成的答案；同时过滤掉输入的部分
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 解码生成的答案
        pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 如果预测答案有效，则与正确答案比较
        if pred and pred[0] in choices:
            cors.append(pred[0] == label)
        # 记录所有预测答案
        all_preds.append(pred.replace("\n", ""))

    # 计算准确率
    acc = np.mean(cors)
    # 打印准确率和结果统计
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print(
        "{} results, {} inappropriate formated answers.".format(
            len(cors), len(all_preds) - len(cors)
        )
    )
    # 返回准确率和所有预测答案
    return acc, all_preds, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="../results/Qwen1.5-7B-Chat")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if is_eval_success(args):
        # eval finished, no need load model anymore, just show the result
        model = None
    else:
        model = init_model(args)

    if "chat" in args.model_name_or_path.lower():
        run_eval(model, tokenizer, eval_chat, args)
    else:
        run_eval(model, tokenizer, eval, args)
