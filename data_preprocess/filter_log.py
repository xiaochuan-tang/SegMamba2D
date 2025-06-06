import os

def filter_log_file(input_log_path, output_log_path):
    """
    读取log文件，删除包含'['字符的行，生成新的log文件。

    :param input_log_path: 原始log文件的路径
    :param output_log_path: 生成的新log文件的路径
    """
    with open(input_log_path, 'r', encoding='utf-8') as infile, \
         open(output_log_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if '[' not in line:
                outfile.write(line)

if __name__ == "__main__":
    input_log_path = "log/seg_hr_mamba/output_v6.log"  # 输入的原始log文件路径
    output_log_path = "log/seg_hr_mamba/filtered_output_v6.log"  # 生成的新log文件路径
    
    # 运行过滤函数
    filter_log_file(input_log_path, output_log_path)
    
    print(f"Filtered log file saved to {output_log_path}")
