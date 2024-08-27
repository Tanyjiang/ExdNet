import numpy as np
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

unloader = transforms.ToPILImage()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


unloader = transforms.ToPILImage()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def PIL_to_Array(PIL):
    img_array = np.array(PIL)
    return img_array


def image_Tensor2ndarray(image_tensor):
    assert (len(image_tensor.shape) == 4 and image_tensor.shape[0] == 1)
    image_tensor = image_tensor.clone().detach()
    image_tensor = image_tensor.to(torch.device('cpu'))
    image_tensor = image_tensor.squeeze()
    image_cv2 = image_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()

    return image_cv2


def PIL_to_Tensor(PIL):
    PIL_array = (np.asarray(PIL) / 255.0)
    Tensor = torch.from_numpy(PIL_array).float().permute(2, 0, 1)

    return Tensor


def compute(img, min_percentile, max_percentile):
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(src, exposure_min, exposure_max=0.99):
    if get_lightness(src) > 130:
        print("not enhance")

    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * exposure_min, 255 * exposure_max, cv2.NORM_MINMAX)

    return out


def get_lightness(src):
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness


def np_save_TensorImg(img_tensor, path):
    img = np.squeeze(img_tensor.cpu().permute(0, 2, 3, 1).numpy())
    im = Image.fromarray(np.clip(img * 255, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


if __name__ == '__main__':
    img = cv2.imread('1.png')
    img = aug(img, 0.033, 0.9)
    cv2.imwrite('out4.png', img)
