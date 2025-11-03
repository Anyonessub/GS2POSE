import os
import shutil

# 设置图片所在的文件夹路径
source_folder = 'data/lmo9/mask_visib'  # 替换为你的文件夹路径
# 设置目标文件夹路径
target_folder = 'data/lmo9/obj_mask_visib'  # 替换为你希望存放新文件夹的路径

# 创建目标文件夹
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.png'):
        # 分割文件名，获取后半部分编号
        part_number = filename.split('_')[1][:6]  # 获取000000到000007
        # 创建对应的物体文件夹
        object_folder = os.path.join(target_folder, part_number)

        if not os.path.exists(object_folder):
            os.makedirs(object_folder)

        # 移动图片到对应的文件夹
        shutil.copy(os.path.join(source_folder, filename), os.path.join(object_folder, filename))

print("图片已成功分类！")