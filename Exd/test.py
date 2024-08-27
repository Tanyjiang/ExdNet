
import torchvision
import torch.optim
import os
import argparse
import numpy as np
from utils import PSNR
from model.Exd_network import Exd
from IQA_pytorch import SSIM
from data_loaders.lsrw import LSRW
from data_loaders.rellisur import RELLISUR
from tqdm import tqdm
import lpips
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--img_val_path', type=str, default='/home/xxx')
# parser.add_argument('--img_val_path', type=str, default='/home/xxx')

parser.add_argument("--dataset", type=str, default='LSRW', help="LSRW or RELLISUR")
parser.add_argument('--result_path', type=str, default='./result_img/')

parser.add_argument('--pre_norm', type=bool, default=True)
config = parser.parse_args()

print(config)
if config.dataset == 'LSRW':
    val_dataset = LSRW(images_path=config.img_val_path, mode='test', normalize=config.pre_norm)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
else:
    val_dataset = RELLISUR(images_path=config.img_val_path, mode='test', normalize=config.pre_norm)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

model = Exd().cuda()
lpips_loss=lpips.LPIPS(net='alex').cuda()
model.load_state_dict(torch.load("/xxx/best_Epoch_Huawei.pth"),strict=False)

model.eval()

ssim = SSIM()
psnr = PSNR()
ssim_list = []

psnr_list = []


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if config.save:
    result_path = config.result_path
    mkdir(result_path)

ssim = SSIM()
psnr = PSNR()
ssim_list = []

psnr_list = []


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if config.save:
    result_path = config.result_path
    mkdir(result_path)

lpips_list = []

with torch.no_grad():
    for i, imgs in tqdm(enumerate(val_loader)):
        low_img, high_img, low_img1, low_img2, low_img3 = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda(), imgs[3].cuda(), imgs[4].cuda()
        low_img_list = [low_img1, low_img2, low_img3]
        low_img = torch.cat([low_img, low_img1, low_img2, low_img3], dim=1)
        mul, add, enhanced_img = model(low_img)
        torchvision.utils.save_image(enhanced_img, result_path + str(i) + '.png')

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        lpips_value = lpips_loss(enhanced_img, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)
        lpips_list.append(lpips_value)

SSIM_mean = np.mean(ssim_list)
PSNR_mean = np.mean(psnr_list)
LPIPS_mean = np.mean(lpips_list)

print('The SSIM Value is:', SSIM_mean)
print('The PSNR Value is:', PSNR_mean)
print('The LPIPS Value is:', LPIPS_mean)

