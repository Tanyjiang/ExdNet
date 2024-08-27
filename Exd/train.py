import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
from model.Myloss import Discriminator_global
from data_loaders.lsrw import LSRW
from data_loaders.rellisur import RELLISUR
from model.Exd_network import Exd
from IQA_pytorch import SSIM
from utils import PSNR,  validation, LossNetwork
from model.function import *
Tensor = torch.cuda.FloatTensor if True else torch.Tensor
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0)
parser.add_argument('--img_val_path', type=str, default='/home/xxx/')
parser.add_argument('--img_path', type=str, default='/home/xxx/')
parser.add_argument("--dataset", type=str, default='LSRW', help="LSRW or RELLISUR")
parser.add_argument("--normalize", action="store_false", help="Default Normalize in LOL training.")
parser.add_argument('--model_type', type=str, default='s')

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--pretrain_dir', type=str, default=None)

parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="./snapshots_folder")
parser.add_argument("--lr_G", type=float, default=5e-4, help="adam: learning rate")
parser.add_argument("--lr_D", type=float, default=1e-4, help="adam: learning rate")

config = parser.parse_args()

print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# Model Setting
model = Exd(type=config.model_type).cuda()
if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))

# Data Setting
if config.dataset == 'LSRW':
    train_dataset = LSRW(images_path=config.img_path, normalize=config.normalize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                                               pin_memory=True)
    val_dataset = LSRW(images_path=config.img_val_path, mode='test', normalize=config.normalize)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
else:
    train_dataset = RELLISUR(images_path=config.img_path, normalize=config.normalize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                                               pin_memory=True)
    val_dataset = RELLISUR(images_path=config.img_val_path, mode='test', normalize=config.normalize)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Loss & Optimizer Setting & Metric
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()

for param in vgg_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

device = next(model.parameters()).device
print('the device is:', device)

L1_loss = nn.L1Loss()
L1_smooth_loss = F.smooth_l1_loss
global_shape = (400, 600)
Global_D = Discriminator_global(input_shape=(3, 400, 600))
Global_D = Global_D.cuda()
Global_D._initialize_weights()
criterion_GAN = torch.nn.MSELoss()
criterion_GAN = criterion_GAN.cuda()
optimizer_Global_D = torch.optim.Adam(Global_D.parameters(), lr=config.lr_D)

Style_loss = InpaintingLoss(VGG16FeatureExtractor())
Style_loss = Style_loss.cuda()

loss_network = LossNetwork(vgg_model)
loss_network.eval()

ssim = SSIM()
psnr = PSNR()
ssim_high = 0
psnr_high = 0

model.train()

for epoch in range(config.num_epochs):

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '   the epoch is:', epoch)
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img, low_img1, low_img2, low_img3 = imgs[0].cuda(), imgs[1].cuda(), imgs[2].cuda(), imgs[
            3].cuda(), imgs[4].cuda()
        low_img_list = [low_img1, low_img2, low_img3]
        low_img = torch.cat([low_img, low_img1, low_img2, low_img3], dim=1)

        optimizer.zero_grad()
        model.train()
        mul, add, enhance_img = model(low_img)

        loss_style, loss_smoth = Style_loss(enhance_img, high_img)

        valid_global = Variable(Tensor(np.ones((high_img.size(0), *Global_D.output_shape))),
                                requires_grad=False)

        gen_validity_global = Global_D(enhance_img)
        loss_GAN_global = criterion_GAN(gen_validity_global, valid_global)

        loss = L1_smooth_loss(enhance_img, high_img) + 0.04 * loss_network(enhance_img, high_img) + (
            1e-4) * loss_GAN_global + 10 * loss_style + (1e-1) * loss_smoth

        loss.backward()

        optimizer.step()
        scheduler.step()
        if ((iteration + 1) % config.display_iter) == 0:
            print("Loss at iteration", iteration + 1, ":", loss.item())

    # Evaluation Model
    model.eval()
    PSNR_mean, SSIM_mean = validation(model, val_loader)

    with open(config.snapshots_folder + '/log.txt', 'a+') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + ' epoch' + str(
            epoch) + ':' + 'the SSIM is ' + str(SSIM_mean) + 'the PSNR is ' + str(PSNR_mean) + '\n')

    if SSIM_mean > ssim_high:
        ssim_high = SSIM_mean
        print('the highest SSIM value is:', str(ssim_high))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))

    f.close()
