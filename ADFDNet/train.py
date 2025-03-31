import os, time, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from utils import read_img, chw_to_hwc, hwc_to_chw
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image

from utils import AverageMeter
from dataset.loader import Real, Syn
from model.cbdnet import Network, fixed_loss


parser = argparse.ArgumentParser(description = 'Train')
parser.add_argument('--bs', default=12, type=int, help='batch size')
parser.add_argument('--ps', default=64, type=int, help='patch size')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--epochs', default=300, type=int, help='sum of epochs')
args = parser.parse_args()


def train(train_loader, model, criterion, optimizer):
	losses = AverageMeter()
	model.train()

	for (noise_img, clean_img, sigma_img, flag) in train_loader:
		input_var = noise_img.cuda()
		target_var = clean_img.cuda()
		sigma_var = sigma_img.cuda()
		flag_var = flag.cuda()

		output = model(input_var)

		loss = criterion(output, target_var, sigma_var, flag_var)
		losses.update(loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
	return losses.avg

def validation(clear_img_path, hazy_img_path, model):
	model.eval()
	SSIM_list = []
	PSNR_list = []
	clear_img_names = os.listdir(clear_img_path)
	hazy_img_names = os.listdir(hazy_img_path)

	for i in range(len(hazy_img_names)):
		hazy_img = cv2.imread(os.path.join(hazy_img_path, hazy_img_names[i]), 0)
		hazy_img = np.expand_dims(hazy_img, axis=-1)
		hazy_img = hazy_img[:, :, ::-1] / 255.0
		hazy_img = np.array(hazy_img).astype('float32')
		input_var = torch.from_numpy(hwc_to_chw(hazy_img)).unsqueeze(0).cuda()
		with torch.no_grad():
			output = model(input_var)

		output_image = chw_to_hwc(output[0, ...].cpu().numpy())
		hazy_img = np.uint8(np.round(np.clip(output_image, 0, 1) * 255.))[:, :, ::-1]

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

	avg_ssim = sum(SSIM_list) / len(SSIM_list)
	avg_psnr = sum(PSNR_list) / len(PSNR_list)
	return avg_ssim, avg_psnr



if __name__ == '__main__':
	save_dir = './save_model/'
	log_file = os.path.join(save_dir, 'log.txt')  # 日志文件路径
	clear_img_path = 'E:/NUC/mynet/data/testsets\labels_1/'
	hazy_img_path = 'E:/NUC/mynet/data/testsets\images_1/'

	model = Network()
	model.cuda()
	model = nn.DataParallel(model)

	if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
		# load existing model
		model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
		print('==> loading existing model:', os.path.join(save_dir, 'checkpoint.pth.tar'))
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		scheduler.load_state_dict(model_info['scheduler'])
		cur_epoch = model_info['epoch']
	else:
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		# create model
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
		cur_epoch = 0
		
	criterion = fixed_loss()
	criterion.cuda()

	train_dataset = Real('./mydata/real_train/', 1, args.ps) + Syn('./mydata/syn_train/', 1, args.ps)
	# train_dataset =  Real('./mydata/Ori_train/', 1, args.ps)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

	# 初始化日志文件
	if not os.path.exists(log_file):
		with open(log_file, 'w') as f:
			f.write("Epoch\tAvg_SSIM\tAvg_PSNR\n")

	for epoch in range(cur_epoch, args.epochs + 1):
		loss = train(train_loader, model, criterion, optimizer)
		scheduler.step()

		avg_ssim, avg_psnr = 0, 0  # 默认值
		if epoch % 5 == 0:
			torch.save({
				'epoch': epoch ,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict()},
				os.path.join(save_dir, f'checkpoint_{epoch :04d}.pth.tar'))
			avg_ssim, avg_psnr = validation(clear_img_path, hazy_img_path, model)
			print("average SSIM", avg_ssim)
			print("average PSNR", avg_psnr)
			# 记录日志
			with open(log_file, 'a') as f:
				f.write(f"{epoch}\t{avg_ssim:.5f}\t{avg_psnr:.5f}\n")

		print('Epoch [{0}]\t'
			'lr: {lr:.6f}\t'
			'Loss: {loss:.5f}'
			.format(
			epoch,
			lr=optimizer.param_groups[-1]['lr'],
			loss=loss))


