import os
import shutil


def select_and_copy_exr_files(source_folder, destination_folder_b, destination_folder_c, step=8):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_folder_b):
        os.makedirs(destination_folder_b)
    if not os.path.exists(destination_folder_c):
        os.makedirs(destination_folder_c)

    # 获取源文件夹中的所有文件
    all_files = os.listdir(source_folder)

    # 仅选择.exr文件
    exr_files = [file for file in all_files if file.endswith('.exr')]

    # 按照每8个文件选择最后一个
    selected_files = [exr_files[i + step - 1] for i in range(0, len(exr_files), step) if i + step - 1 < len(exr_files)]

    # 剩余的文件
    remaining_files = [file for file in exr_files if file not in selected_files]

    for file in selected_files:
        # 生成源文件路径和目标文件路径
        source_path = os.path.join(source_folder, file)
        destination_path_b = os.path.join(destination_folder_b, file)

        # 复制文件到B文件夹
        shutil.copy(source_path, destination_path_b)

    for file in remaining_files:
        # 生成源文件路径和目标文件路径
        source_path = os.path.join(source_folder, file)
        destination_path_c = os.path.join(destination_folder_c, file)

        # 复制文件到C文件夹
        shutil.copy(source_path, destination_path_c)

    print(f"总共选择并复制了{len(selected_files)}个文件到{destination_folder_b}")
    print(f"剩余的{len(remaining_files)}个文件已复制到{destination_folder_c}")


# 调用函数，设置源文件夹和目标文件夹路径
source_folder = 'F:/6D/exr/'  # 替换为你的A文件夹路径
destination_folder_b = 'F:/6D/GS_seg/NOCS_data/val/exr/'  # 替换为你的B文件夹路径
destination_folder_c = 'F:/6D/GS_seg/NOCS_data/train/exr/'  # 替换为你的C文件夹路径

select_and_copy_exr_files(source_folder, destination_folder_b, destination_folder_c)

