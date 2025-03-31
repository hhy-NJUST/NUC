# # 定义目录路径
# directory1 = '../mydata/syn_train/images_no/'
# directory2 = '../mydata/syn_train/02/'
#
# # 遍历目录1下的所有文件
# for filename in os.listdir(directory1):
#     if filename.endswith('.bmp') or filename.endswith('.png'):
#         # 读取图片
#         img = Image.open(os.path.join(directory1, filename))
#
#         # 在文件名后加上'_0'后缀
#         new_filename = filename.split('.')[0] + '_noisy.' + filename.split('.')[-1]
#
#         # 保存重命名后的图片到目录2
#         img.save(os.path.join(directory2, new_filename))
#
# print("图片重命名完成！")

# 定义目录路径
import os
from PIL import Image
import shutil

# 定义目录路径
# directory =  '../mydata/syn_data/05/'
#
# # 遍历目录下的所有文件
# for filename in os.listdir(directory):
#     # 检查文件是否以 '.bmp' 结尾
#     if filename.endswith('.bmp'):
#         # 构建旧文件路径和新文件路径
#         old_filepath = os.path.join(directory, filename)
#         new_filename = os.path.splitext(filename)[0] + '.png'  # 将后缀 '.bmp' 替换为 '.png'
#         new_filepath = os.path.join(directory, new_filename)
#
#         # 重命名文件
#         os.rename(old_filepath, new_filepath)
#         print(f'Renamed {filename} to {new_filename}')



#
# import os
# import shutil
#
# # 源文件夹路径
# original_folder = "../mydata/syn_data/05/" # "../mydata/syn_data/05/", "../mydata/syn_data/02/"
#
# # 目标文件夹路径
# new_folder = "../mydata/syn_data/mix/"
#
# # 创建目标文件夹
# if not os.path.exists(new_folder):
#     os.makedirs(new_folder)
#
# # 遍历原始文件夹中的所有文件
# for filename in os.listdir(original_folder):
#     if filename.endswith('.png'):  # 确保文件是以 .png 结尾的图片文件
#         # 获取文件名和后缀
#         name, ext = os.path.splitext(filename)
#         # 获取原始编号
#         original_number = int(name.split('_')[0])
#         # 计算新的编号
#         new_number = original_number + 212
#         # 构造新的文件名
#         new_name = f"{new_number:04d}{name[4:]}{ext}"
#
#         new_path = os.path.join(new_folder, new_name)
#         # 拷贝文件到新的文件夹中
#         shutil.copyfile(os.path.join(original_folder, filename), new_path)
#         print(f'Renamed {filename} to {new_name}')

import os
import cv2
import numpy as np


def gabor_filter_feature_extraction(image, theta, sigma, lambda_, gamma, phi):
    # 转换图像为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 初始化 Gabor 滤波器
    kernel = cv2.getGaborKernel((sigma, sigma), lambda_, theta, sigma, gamma, phi)

    # 使用滤波器对图像进行卷积
    filtered_image = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)

    return filtered_image

def togray():
    # 定义目录路径
    input_dir = "./mydata/test/3/"
    output_dir = "./mydata/test/"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 拼接输入文件路径
        input_path = os.path.join(input_dir, filename)

        # 读取图像
        image = cv2.imread(input_path)

        # 将图像转换为单通道灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 拼接输出文件路径
        output_path = os.path.join(output_dir, filename)

        # 保存单通道灰度图
        cv2.imwrite(output_path, gray_image)

        print(f'{filename} 转换完成，并保存到 {output_path}')


if __name__ == '__main__':
    togray()