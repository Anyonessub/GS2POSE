import cv2
import numpy as np
import os
# from utils.my_utils import MyData, unet
# import torch

def move_red_object_to_center(image):
    # 假设物体是图像中最大的轮廓
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        object_center_x = x + w // 2
        object_center_y = y + h // 2

        image_center_x = image.shape[1] // 2
        image_center_y = image.shape[0] // 2

        move_x = image_center_x - object_center_x
        move_y = image_center_y - object_center_y

        translation_matrix = np.float32([[1, 0, move_x], [0, 1, move_y]])
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

        return translated_image, (move_x, move_y)
    else:
        raise ValueError("No red object found in the image")


def center_crop(image, crop_size=256):
    height, width = image.shape[:2]
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    return image[start_y:start_y + crop_size, start_x:start_x + crop_size]


def nocs(image):
    # Placeholder function for NOCS processing
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust the hue of the red object to add some variety
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 60) % 180

    colored_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return colored_image


def reverse_translation(image, move_x, move_y):
    translation_matrix = np.float32([[1, 0, -move_x], [0, 1, -move_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return translated_image


def resize_to_original(image, original_size):
    # 获取裁剪后的图像尺寸
    cropped_height, cropped_width = image.shape[:2]

    # 创建与原始图像大小相同的黑色背景
    background = np.zeros((original_size[0], original_size[1], 3), dtype=np.uint8)

    # 计算要将裁剪后的图像放置在背景中的位置
    start_x = (original_size[1] - cropped_width) // 2
    start_y = (original_size[0] - cropped_height) // 2

    # 将裁剪后的图像放置到背景上
    background[start_y:start_y + cropped_height, start_x:start_x + cropped_width] = image

    return background

def add_black_border(image, border_size=20):
    height, width = image.shape[:2]
    black_border = np.zeros_like(image)
    black_border[border_size:height-border_size, border_size:width-border_size] = image[border_size:height-border_size, border_size:width-border_size]

    return black_border

def process_images_in_folder(input_folder, output_folder,nocs_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #加载模型
    classes = 3
    # G = unet(num_classes=classes, pretrained=False, backbone="resnet50").cuda()  # generator model
    # checkpoint_path = 'checkpoints_2/checkpoint-1315_11.pth'
    # checkpoint = torch.load(checkpoint_path)
    # G.load_state_dict(checkpoint["net"], strict=True)
    # G.eval()


    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            try:
                original_size = image.shape[:2]

                # Step 1: 将红色物体移动到画面中心，其中translated_image为移动到画面中心的图片,translation为移动的距离
                translated_image, translation = move_red_object_to_center(image)

                # Step 2: 将图片裁剪为256*256大小
                cropped_image = center_crop(translated_image, 256)

                # 加载nocs图片
                output_image_path = os.path.join(nocs_folder, filename)
                processed_image = cv2.imread(output_image_path)

                #边缘赋予黑色
                processed_image = add_black_border(processed_image)

                # Step 4: Resize to the original image size
                resized_image = resize_to_original(processed_image, original_size)

                # Step 5: Reverse the translation to move the object back to its original position
                final_image = reverse_translation(resized_image, *translation)


                # Step 6: Create mask to retain only non-black regions from original image
                non_black_mask = cv2.inRange(image, (1, 1, 1), (255, 255, 255))
                final_image = cv2.bitwise_and(final_image, final_image, mask=non_black_mask)

                # Save the final image
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, final_image)
                print(f"Processed and saved: {output_path}")
            except ValueError as e:
                print(f"Skipping {filename}: {e}")


# 设置输入和输出文件夹路径
input_folder = 'lm_data/000014/image'
output_folder = 'lm_data/000014/result'
nocs_folder = 'lm_data/000014/output'

# 处理文件夹中的所有图像
process_images_in_folder(input_folder, output_folder,nocs_folder)



