import os
import pdb

def rename_files(directory):
    # 遍历指定目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否符合特定格式
        if filename.startswith("ops_0_") and (filename.endswith(".e") or filename.endswith(".o")):
            # 分割文件名以提取数字部分
            parts = filename.split('_')
            if len(parts) > 2 and parts[2].split(".")[0].isdigit():
                # 计算新的数字部分
                new_number = int(parts[2].split(".")[0])
                # 构建新的文件名
                new_filename = f"halfcheetah_medium_replay_{parts[1]}_{new_number}{os.path.splitext(filename)[1]}"
                # 完整的文件路径
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_filename)
                # 重命名文件
                os.rename(old_file, new_file)
                print(f"Renamed '{old_file}' to '{new_file}'")

# 指定要批量重命名文件的目录
directory = '/home/xiaoan/OPE4RL/d3rlpy/.onager/logs/gaoqitong-exxact'
rename_files(directory)
