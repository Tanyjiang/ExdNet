import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
from data_loaders.tes import PIL_to_Array, PIL_to_Tensor, aug
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
random.seed(1143)

def populate_train_list_lsrw(images_path, mode='train'):
    img_path=images_path.replace('Huawei', 'Nikon')
    image_list_lowlight = glob.glob(images_path + '*.jpg')+glob.glob(img_path+ '*.jpg')
    train_list = image_list_lowlight
    if mode == 'train':
        random.shuffle(train_list)
    return train_list

def populate_train_list(images_path, mode='train'):
    image_list_lowlight = glob.glob(images_path + '*.jpg')
    train_list = image_list_lowlight
    if mode == 'train':
        random.shuffle(train_list)

    return train_list


def mulit_exposure(image):
    raw_img = PIL_to_Tensor(image)
    image = PIL_to_Array(image)
    img1 = aug(image, 0.033, 0.9)
    img2 = aug(image, 0.04, 0.9)
    img3 = aug(image, 0.1, 1)

    return img1, img2, img3


class LSRW(data.Dataset):
    def __init__(self, images_path, mode='train', normalize=True):
        self.train_list = populate_train_list(images_path, mode)
        self.mode = mode
        self.data_list = self.train_list
        self.normalize = normalize
        print("Total examples:", len(self.train_list))

    def FLIP_LR(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high

    def FLIP_UD(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        return low, high

    def get_params(self, low):
        self.w, self.h = low.size

        self.crop_height = random.randint(self.h / 2, self.h)
        self.crop_width = random.randint(self.w / 2, self.w)

        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i, j

    def Random_Crop(self, low, high):
        self.i, self.j = self.get_params((low))
        if random.random() > 0.5:
            low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return low, high

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        if self.mode == 'train':
            data_lowlight = Image.open(data_lowlight_path)
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high').replace('low', 'high'))
            data_lowlight, data_highlight = self.FLIP_LR(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.FLIP_UD(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)
            data_lowlight = data_lowlight.resize((600, 400), Image.ANTIALIAS)
            data_highlight = data_highlight.resize((600, 400), Image.ANTIALIAS)
            data_lowlight1, data_lowlight2, data_lowlight3 = mulit_exposure(data_lowlight)
            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)
            data_lowlight1, data_lowlight2, data_lowlight3 = (np.asarray(data_lowlight1) / 255.0), (
                    np.asarray(data_lowlight2) / 255.0), (np.asarray(data_lowlight3) / 255.0)

            if self.normalize:
                transform_input = Compose(
                    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
                transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
                return transform_input(data_lowlight), transform_gt(data_highlight), transform_input(
                    data_lowlight1), transform_input(data_lowlight2), transform_input(data_lowlight3)
            else:
                data_lowlight, data_highlight, data_highlight1, data_highlight2, data_highlight3 = torch.from_numpy(
                    data_lowlight).float(), torch.from_numpy(data_highlight).float(), torch.from_numpy(
                    data_lowlight1).float(), torch.from_numpy(data_lowlight2).float(), torch.from_numpy(
                    data_lowlight3).float()
                return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1), data_lowlight1.permute(2, 0,
                                                                                                               1), data_lowlight2.permute(
                    2, 0, 1), data_lowlight3.permute(2, 0, 1)

        elif self.mode == 'test':
            data_lowlight = Image.open(data_lowlight_path)
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high').replace('low', 'high'))
            data_lowlight = data_lowlight.resize((600, 400), Image.ANTIALIAS)
            data_highlight = data_highlight.resize((600, 400), Image.ANTIALIAS)
            data_lowlight1, data_lowlight2, data_lowlight3 = mulit_exposure(data_lowlight)
            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)
            data_lowlight1, data_lowlight2, data_lowlight3 = (np.asarray(data_lowlight1) / 255.0), (
                    np.asarray(data_lowlight2) / 255.0), (np.asarray(data_lowlight3) / 255.0)
            if self.normalize:
                transform_input = Compose(
                    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
                transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
                return transform_input(data_lowlight), transform_gt(data_highlight), transform_input(
                    data_lowlight1), transform_input(data_lowlight2), transform_input(data_lowlight3)
            else:
                data_lowlight, data_highlight, data_highlight1, data_highlight2, data_highlight3 = torch.from_numpy(
                    data_lowlight).float(), torch.from_numpy(data_highlight).float(), torch.from_numpy(
                    data_lowlight1).float(), torch.from_numpy(data_lowlight2).float(), torch.from_numpy(
                    data_lowlight3).float()
                return data_lowlight.permute(2, 0, 1), data_highlight.permute(2, 0, 1), data_lowlight1.permute(2, 0,
                                                                                                               1), data_lowlight2.permute(
                    2, 0, 1), data_lowlight3.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
