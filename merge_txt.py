import os

def concatenate_txt_files_in_folder(folder_path):
    """
    Concatenates all .txt files found directly in the given folder into a single file with the same name.
    
    Parameters:
    folder_path (str): The path to the folder containing .txt files.
    
    Returns:
    None
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"The provided path '{folder_path}' is not a directory.")
    
    # 获取文件夹名作为新文件的名字
    folder_name = os.path.basename(folder_path)
    new_file_path = os.path.join(os.path.dirname(folder_path), f"{folder_name}.txt")
    
    combined_content = ""
    
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                combined_content += f"--- Content from {file} ---\n{content}\n\n"
    
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    print(f"All .txt files in '{folder_path}' have been concatenated and saved to '{new_file_path}'.")

if __name__ == "__main__":
    folder_path = './results_bio/Qwen2-7B-Instruct-zeroshot-finetuned-v2_0_shot'
    concatenate_txt_files_in_folder(folder_path)
