import cv2
import numpy as np
import os


def move_object_to_center(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU)
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

        return translated_image
    else:
        raise ValueError("No object found in the image")


def crop_image_center(image, crop_size=256):
    height, width = image.shape[:2]
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    return image[start_y:start_y + crop_size, start_x:start_x + crop_size]


def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            try:
                translated_image = move_object_to_center(image)
                cropped_image = crop_image_center(translated_image)

                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_image)
                print(f"Processed and saved: {output_path}")
            except ValueError as e:
                print(f"Skipping {filename}: {e}")


# 设置输入和输出文件夹路径
input_folder = 'lm_data/000014/image'  # 替换为你的输入图片文件夹路径
output_folder = 'lm_data/000014/gt'  # 替换为你的输出图片文件夹路径

# input_folder = 'data/000014/val/gt'  # 替换为你的输入图片文件夹路径
# output_folder = 'data/000014/val/gt'  # 替换为你的输出图片文件夹路径

# 处理文件夹中的所有图像
process_images_in_folder(input_folder, output_folder)