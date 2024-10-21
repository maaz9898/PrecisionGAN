import cv2
import numpy as np
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import torchvision.models as models
from skimage import color
from pytorch_msssim import ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_WIDTH = 256
IMG_HEIGHT = 256

def read_image(image):
        image = np.array(image)
        width = image.shape[1]
        width_half = width // 2
       
        input_image = image[:, :width_half, :]
        target_image = image[:, width_half:, :]
    
        input_image = input_image.astype(np.float32)
        target_image = target_image.astype(np.float32)
    
        return input_image, target_image

def random_crop(image, dim):
    height, width, _ = dim
    x, y = np.random.uniform(low=0,high=int(height-256)), np.random.uniform(low=0,high=int(width-256))  
    return image[:, int(x):int(x)+256, int(y):int(y)+256]
    
def random_jittering_mirroring(input_image, target_image, height=286, width=286):
    #resizing to 286x286
    input_image = cv2.resize(input_image, (height, width) ,interpolation=cv2.INTER_NEAREST)
    target_image = cv2.resize(target_image, (height, width),
                               interpolation=cv2.INTER_NEAREST)
    
    #cropping (random jittering) to 256x256
    stacked_image = np.stack([input_image, target_image], axis=0)
    cropped_image = random_crop(stacked_image, dim=[IMG_HEIGHT, IMG_WIDTH, 3])
    
    input_image, target_image = cropped_image[0], cropped_image[1]
    if torch.rand(()) > 0.5:
     # random mirroring
        input_image = np.fliplr(input_image)
        target_image = np.fliplr(target_image)
    return input_image, target_image
    
def normalize(inp, tar):
    input_image = (inp / 127.5) - 1
    target_image = (tar / 127.5) - 1
    return input_image, target_image
        
class Train_Normalize(object):
    def __call__(self, image):
        inp, tar = read_image(image)
        inp, tar = random_jittering_mirroring(inp, tar)
        inp, tar = normalize(inp, tar)
        image_a = torch.from_numpy(inp.copy().transpose((2,0,1)))
        image_b = torch.from_numpy(tar.copy().transpose((2,0,1)))
        return image_a, image_b    
    
class Val_Normalize(object):
    def __call__(self, image):
        inp, tar = read_image(image)
        #inp, tar = random_jittering_mirroring(inp, tar)
        inp, tar = normalize(inp, tar)
        image_a = torch.from_numpy(inp.copy().transpose((2,0,1)))
        image_b = torch.from_numpy(tar.copy().transpose((2,0,1)))
        return image_a, image_b
    
def weights_init(net, init_type='normal', scaling=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_norm_layer(norm_type = 'instance'):
    """Return a normalization layer"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    return norm_layer

adversarial_loss = nn.BCELoss() 

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:16].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, generated_image, target_img):
        gen_features = self.vgg(generated_image)
        target_features = self.vgg(target_img)
        perceptual_loss = self.criterion(gen_features, target_features)
        return perceptual_loss

def rgb_to_lab(image):
    """Convert a batch of images from RGB to Lab color space.
    
    Args:
        image (Tensor): a batch of images in RGB, shape (N, C, H, W)

    Returns:
        Tensor: a batch of images in Lab, shape (N, C, H, W)
    """
    # Detach the tensor and move it to CPU
    image_np = image.detach().cpu().numpy().transpose(0, 2, 3, 1)  # Convert from (N, C, H, W) to (N, H, W, C)
    image_lab_np = np.array([color.rgb2lab(img) for img in image_np])
    image_lab = torch.from_numpy(image_lab_np).float().to(image.device)
    image_lab = image_lab.permute(0, 3, 1, 2)  # Convert back to (N, C, H, W)
    return image_lab

def color_consistency_loss(generated_image, target_img):
    generated_image_lab = rgb_to_lab(generated_image)
    target_img_lab = rgb_to_lab(target_img)
    l_loss = F.mse_loss(generated_image_lab[:, 0, :, :], target_img_lab[:, 0, :, :])
    a_loss = F.mse_loss(generated_image_lab[:, 1, :, :], target_img_lab[:, 1, :, :])
    b_loss = F.mse_loss(generated_image_lab[:, 2, :, :], target_img_lab[:, 2, :, :])
    return l_loss + 10 * a_loss + 10 * b_loss  # More weight to color channels

def ssim_loss(generated_image, target_img):
    ssim_loss_value = 1 - ssim(generated_image, target_img, data_range=1, size_average=True)  # data_range depends on the range of your input, size_average for a mean over all images in batch
    return ssim_loss_value

# perceptual_loss_module = PerceptualLoss(device)

# Binary Cross-Entropy Loss for Generator
def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)

    # l1_l = F.l1_loss(generated_image, target_img)
    MSE_1 = F.mse_loss(generated_image, target_img)
    # perceptual_loss = perceptual_loss_module(generated_image, target_img)  # Use the existing instance
    # color_loss = color_consistency_loss(generated_image, target_img)
    ssim_l = ssim_loss(generated_image, target_img)

    # gen_total_loss = gen_loss + 100 * l1_l + 100 * MSE_1 + perceptual_loss + color_loss + 10 * ssim_l

    # gen_total_loss = gen_loss + 100*MSE_1 + perceptual_loss

    gen_total_loss = gen_loss + 100*MSE_1 + 10*ssim_l

    # print("gen_loss: ", gen_loss, "l1_l: ", l1_l, "MSE_1: ", MSE_1, "perceptual_loss: ", perceptual_loss, "color_loss: ", color_loss, "ssim_l: ", ssim_l)

    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss