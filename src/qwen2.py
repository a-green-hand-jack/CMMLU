import os
import torch
import numpy as np
import argparse
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from peft import PeftModel

import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 7925))
#     print("Waiting for debugger attach")
#     print("the python code is qwen2.py")
#     print("the host is: localhost, the port is: 7925")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# 设置HF_HOME环境变量
os.environ["HF_HOME"] = "/root/autodl-fs/pre-trained-models/"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = "cuda"  # the device to load the model onto


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
        cache_dir=args.cache_dir,
    )

    # Import the generation configuration of the pre-trained model to ensure the model's generation behavior is consistent with the pre-trained model.
    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    return model


def init_fine_tuned_model(args):
    """
    初始化微调后的模型。

    该函数从预训练模型开始，对其进行微调以适应特定任务。它使用了AutoModelForCausalLM来加载预训练模型，
    然后配置生成设置，并最终加载特定的微调模型。

    参数:
    args: Namespace
        包含模型配置和路径的参数对象。其中包括模型名称或路径（model_name_or_path）、缓存目录（cache_dir）和PEFT模型ID（peft_model_id）。

    返回:
    PeftModel
        微调后的预训练模型，配置为适用于特定任务的生成模型。
    """

    # 从预训练模型加载AutoModelForCausalLM，配置包括模型路径、信任远程代码、设备映射、数据类型和缓存目录。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
    )

    # 设置模型的生成配置，从预训练模型加载并信任远程代码。
    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    # 从预训练模型加载PEFT模型，指定微调后的模型和模型ID。
    pretrained_model = PeftModel.from_pretrained(
        model=model, model_id=args.peft_model_id
    )
    print(pretrained_model)

    # 返回微调后的PEFT模型。
    return pretrained_model

def extract_answers(text: str) -> str:
    """
    从给定的字符串中提取大写的英文字母，并返回一个新的字符串。

    参数:
    text (str): 包含大写字母的原始字符串。

    返回:
    str: 仅包含从原始字符串中提取的大写字母的新字符串。
    """

    # 提取并返回所有大写字母
    return ''.join([char for char in text if char.isupper()])



def is_composed_of(pred, choices):
    """
    检查一个字符串是否完全由另一组字符串中的字符组成。

    参数:
    pred (str): 需要检查的字符串。
    choices (iterable): 作为检查依据的字符串集合。

    返回:
    bool: 如果pred中的所有字符都存在于choices中，则返回True，否则返回False。

    示例:
    >>> is_composed_of("abc", ["a", "b", "c", "d"])
    True
    >>> is_composed_of("xyz", ["a", "b", "c"])
    False
    """

    # 使用all函数检查pred中的每个字符是否都在choices中
    # 这里使用生成器表达式提供一个更高效和简洁的方法来检查字符存在性
    return all(char in choices for char in pred)


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
        pred_model = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        pred = extract_answers(text=pred_model)
        
        # 如果预测答案有效，则与正确答案比较
        if pred and is_composed_of(pred, choices):
            cors.append(pred == label)

        all_preds.append(pred.replace("\n", ""))

    # 计算准确率
    acc = np.mean(cors)
    # 将准确率写入本地文件
    out_file = os.path.join(args.save_dir, f"results_{subject}.txt")
    with open(out_file, "a") as file:
        file.write(f"Accuracy for {subject}: {acc:.3f}\n")
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
    parser.add_argument(
        "--cache_dir", type=str, default="/root/autodl-fs/pre-trained-models/hub/"
    )
    parser.add_argument("--peft_model_id", type=str, default="", help="peft model id")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if is_eval_success(args):
        # eval finished, no need load model anymore, just show the result
        model = None
    else:
        if args.peft_model_id:
            model = init_fine_tuned_model(args)
        else:
            model = init_model(args)

    run_eval(model, tokenizer, eval_chat, args)
