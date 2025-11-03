import os
from PIL import Image

# 定义文件夹路径
gs_image_folder = 'data/lmo/image'  # gs_image 文件夹路径
mask_folder = 'data/lmo/mask_visib'            # mask 文件夹路径
output_folder = 'data/lmo/image2'        # 输出文件夹路径

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取并排序文件夹中的图片文件名
gs_images = sorted(os.listdir(gs_image_folder))
mask_images = sorted(os.listdir(mask_folder))

# 确保两个文件夹中的文件数量一致
if len(gs_images) != len(mask_images):
    print("两个文件夹中的图片数量不一致，请检查！")
else:
    # 遍历图片
    for gs_image_name, mask_image_name in zip(gs_images, mask_images):
        if gs_image_name.endswith(('.png', '.jpg', '.jpeg')) and mask_image_name.endswith(('.png', '.jpg', '.jpeg')):
            # 构建图片路径
            gs_image_path = os.path.join(gs_image_folder, gs_image_name)
            mask_image_path = os.path.join(mask_folder, mask_image_name)

            # 打开 gs_image 和 mask 图片
            gs_image = Image.open(gs_image_path).convert('RGBA')
            mask_image = Image.open(mask_image_path).convert('RGBA')

            # 创建一个新的图片用于保存结果
            result_image = Image.new('RGBA', gs_image.size)

            # 遍历每个像素
            for x in range(gs_image.width):
                for y in range(gs_image.height):
                    gs_pixel = gs_image.getpixel((x, y))
                    mask_pixel = mask_image.getpixel((x, y))

                    # 如果 mask 像素是黑色 (0, 0, 0, 255)
                    if mask_pixel[0] == 0 and mask_pixel[1] == 0 and mask_pixel[2] == 0:
                        # 将 gs_image 中的像素设置为黑色
                        result_image.putpixel((x, y), (0, 0, 0, 255))
                    else:
                        # 否则保留原始 gs_image 像素
                        result_image.putpixel((x, y), gs_pixel)

            # 保存结果图片，使用 mask_image_name 作为文件名
            result_image.save(os.path.join(output_folder, mask_image_name))

print("处理完成！")