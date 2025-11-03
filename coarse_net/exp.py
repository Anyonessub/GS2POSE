import os

def rename_images(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名中是否包含 "_000000"
        if "_000000" in filename:
            # 构造新的文件名
            new_filename = filename.replace("_000000", "")
            # 获取完整路径
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

# 指定文件夹路径
folder_path = "0001/image"

# 调用函数
rename_images(folder_path)


