import os, time, scipy.io, shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import time
from model.cbdnet import Network
from utils import read_img, chw_to_hwc, hwc_to_chw
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
# parser = argparse.ArgumentParser(description = 'Test')
# parser.add_argument('input_filename', type=str)
# parser.add_argument('output_filename', type=str)
# args = parser.parse_args()

def validation(clear_img_path, hazy_img_path, model, output_dir, output_file):
	model.eval()
	SSIM_list = []
	PSNR_list = []
	inference_times = []
	clear_img_names = os.listdir(clear_img_path)
	hazy_img_names = os.listdir(hazy_img_path)

	with open(output_file, 'w') as f:
		for i in range(len(hazy_img_names)):
			hazy_img = cv2.imread(os.path.join(hazy_img_path, hazy_img_names[i]), 0)
			hazy_img = np.expand_dims(hazy_img, axis=-1)
			hazy_img = hazy_img[:, :, ::-1] / 255.0
			hazy_img = np.array(hazy_img).astype('float32')
			input_var = torch.from_numpy(hwc_to_chw(hazy_img)).unsqueeze(0).cuda()
			with torch.no_grad():
				start_time = time.time()
				output = model(input_var)
				end_time = time.time()
				inference_time = end_time - start_time
				inference_times.append(inference_time)

			output_image = chw_to_hwc(output[0, ...].cpu().numpy())
			hazy_img = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[:, :, ::-1]
			cv2.imwrite(os.path.join(output_dir, hazy_img_names[i]), hazy_img)

			clear_img = cv2.imread(os.path.join(clear_img_path, clear_img_names[i]), 0)
			clear_img = np.expand_dims(clear_img, axis=-1)

			if clear_img.shape[0] != hazy_img.shape[0] or clear_img.shape[1] != hazy_img.shape[1]:
				pil_img = Image.fromarray(hazy_img)
				pil_img = pil_img.resize((clear_img.shape[1], clear_img.shape[0]))  # 和clear_img的宽和高保持一致
				hazy_img = np.array(pil_img)

		# 计算PSNR
			PSNR = peak_signal_noise_ratio(clear_img, hazy_img)
			print(i + 1, 'PSNR: ', PSNR)
			PSNR_list.append(PSNR)

		# 计算SSIM
			SSIM = structural_similarity(clear_img, hazy_img, channel_axis=2)
			print(i + 1, 'SSIM: ', SSIM)
			SSIM_list.append(SSIM)
			f.write(f"{i + 1} PSNR: {PSNR}\n")
			f.write(f"{i + 1} SSIM: {SSIM}\n")

	with open(output_file, 'a') as f:
		avg_ssim = sum(SSIM_list) / len(SSIM_list)
		avg_psnr = sum(PSNR_list) / len(PSNR_list)
		avg_inference_time = sum(inference_times) / len(inference_times)
		print("average SSIM", avg_ssim)
		print("average PSNR", avg_psnr)
		f.write(f"average SSIM: {avg_ssim}\n")
		print("average inference time per image (seconds):", avg_inference_time)
		f.write(f"average PSNR: {avg_psnr}\n")


if __name__ == '__main__':
	# clear_img_path = 'E:/NUC/mynet/data/testsets\labels_1/'
	# hazy_img_path = 'E:/NUC/mynet/data/testsets\images_1/'
	clear_img_path = 'E:/NUC/compare/1D-GF-master/images/'
	hazy_img_path ='E:/NUC/compare/1D-GF-master/images/'
	output_dir = './results/1D-GF-master/images/'  # 替换为你的输出图像文件名
	output_file = './results/1D-GF-master/images/psnr_ssim_results.txt'
	# input_dir = 'E:/NUC/mynet/data/testsets\images_1/'  # 替换为你的输入图像文件名
	# input_dir = './mydata/thermal_test/'  # 替换为你的输入图像文件名
	# save_dir = './save_model/finetune/'
	save_dir = './save_model/final_net/'

	# 创建目标文件夹
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	model = Network()
	model.cuda()
	model = nn.DataParallel(model)
	model.eval()
	if os.path.exists(os.path.join(save_dir, 'checkpoint_0300.pth.tar')):
        # load existing model
		model_info = torch.load(os.path.join(save_dir, 'checkpoint_0300.pth.tar'))
		model.load_state_dict(model_info['state_dict'])
	else:
		print('Error: no trained model detected!')
		exit(1)

	validation(clear_img_path, hazy_img_path, model, output_dir, output_file)






# model = Network()
# model.cuda()
# model = nn.DataParallel(model)
#
# model.eval()
#
# if os.path.exists(os.path.join(save_dir, 'checkpoint_0220.pth.tar')):
#     # load existing model
#     model_info = torch.load(os.path.join(save_dir, 'checkpoint_0220.pth.tar'))
#     model.load_state_dict(model_info['state_dict'])
# else:
#     print('Error: no trained model detected!')
#     exit(1)
#
# input_files = os.listdir(input_dir)
# for input_filename in input_files:
#     # 构建输入和输出文件名的完整路径
#     input_filepath = os.path.join(input_dir, input_filename)
#     output_filename = os.path.join(output_dir, input_filename)
#
#     input_image = read_img(input_filepath)
#     input_var = torch.from_numpy(hwc_to_chw(input_image)).unsqueeze(0).cuda()
#
#     with torch.no_grad():
#         stime = time.time()
#         output = model(input_var)
#         etime = time.time()
#         T = etime - stime
#         print("time:", T)
#
#
#     output_image = chw_to_hwc(output[0, ...].cpu().numpy())
#     output_image = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[:, :, ::-1]
#
#     cv2.imwrite(output_filename, output_image)

