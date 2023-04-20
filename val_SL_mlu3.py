from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch_mlu
import torch.nn.functional as F

#torch.cuda.current_device()
#torch.cuda._initialized = True
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
# from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from model.arch_unet_plus4_6 import UNet
# from model.arch_unet_NAF import NAFNet
# from model.ridnet import RIDNET
from torch.utils.tensorboard import SummaryWriter

# from model.res_unet_plus import ResUnetPlusPlus
# from astropy.visualization import ZScaleInterval, LinearStretch, ImageNormalize
# from model.SRMNet import SRMNet
# from model.mirnet_v2_arch import MIRNet_v2
# from model.network_scunet import SCUNet
# from model.CDLnet import CDLNet
# from warmup_scheduler import GradualWarmupScheduler

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str,
                    default='./dataset/astro_dataset/train_bigastro_png_sub256_useful')  # ./dataset/astro_dataset/train_bigastro_png_sub256_useful
parser.add_argument('--val_dirs', type=str, default='datasets/astro_dataset/val_bigastro_png')
parser.add_argument('--save_model_path', type=str, default='./results_RL_128')
parser.add_argument('--model_weight_path', type=str, default='./weights/epoch_model_100.pth')  # 权重文件路径
parser.add_argument('--log_name', type=str, default='UNet4_6_gauss25_bigastro_b16e100')
parser.add_argument('--gpu_devices', default='0,1', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_feature', type=int, default=48)  # 56
parser.add_argument('--n_channel', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)  # 1e-3   3e-4
parser.add_argument('--gamma', type=float, default=0.5)  # 0.5im.convert('RGB')
parser.add_argument('--n_epoch', type=int, default=100)  # 训练的轮数
parser.add_argument('--n_snapshot', type=int, default=10)  # 每多少轮保存权重并验证集验证
parser.add_argument('--batchsize', type=int, default=32)  # 4   SCUNet,SRMNet:8
parser.add_argument('--patchsize', type=int, default=256)  #
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=2.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("mlu" if 1 else "cpu")


# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.current_device())

def checkpoint(net, epoch, name):  # checkpoint(network, 0, "model") checkpoint(network, epoch, "model")
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)  # 存储训练权重路径
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="mlu")
    g_cuda_generator.manual_seed(operation_seed_counter)  # 。参数的初始化是随机的，为了让每次的结果一致，我们需要设置随机种子
    return g_cuda_generator


class AugmentNoise(object):  # 添加噪声# noise_adder = AugmentNoise(style=opt.noisetype)
    def __init__(self, style):
        print(style)
        if style.startswith('gauss'):  # 判断字符串是否以给定字符开头
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')  # gauss25分割得到25
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"  # 只有一个噪声值
            elif len(self.params) == 2:
                self.style = "gauss_range"  # 噪声值有一个变化范围
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):  # 给训练集添加噪声 noisy = noise_adder.add_train_noise(clean)
        shape = x.shape  # 图像的shape是 [4,3,256,256] batchsize=4
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1), device=x.device)  # n,c,h,w 4维
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):  # 给验证集添加噪声
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.uint8)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.uint8)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.uint8)


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    """
       input: tensor数据，四维， Batchsize, channel, height, width
       kernel_size: 核大小，决定输出tensor的数目。
       dilation: 输出形式是否有间隔，稍后详细讲。
       padding：一般是没有用的必要
       stride： 核的滑动步长。稍后详细讲
   """

    return unfolded_x.view(n, c * block_size ** 2, h // block_size,
                           w // block_size)


def generate_mask_pair(img):  # mask1, mask2 = generate_mask_pair(noisy)#得到相邻的两张图的标签
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape  # w:256 h:256 n:4 c:3
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),  # bool类型的全false tensor向量 128*128*4
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4,),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],  # 四个单元里面随机选两个  分到两个子图
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2,),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2,),  # 128*128
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):  # generate_subimages(noisy, mask1)#生成噪声图像子图1  size[4,3,128,128]
    n, c, h, w = img.shape
    subimage = torch.zeros(n,  # 分成h//2*w//2 个单元  每个单元2*2大小  子图大小128*128
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):  # 每一个维度抽取子图 c=3
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))  # 返回所有文件路径的列表
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):  # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。
        # 当实例对象做P[key]运算时，就会调用类中的__getitem__()方法
        # fetch image
        fn = self.train_fns[index]  # 图片
        im = Image.open(fn)
        # im = cv2.imread(fn)
        #    np.set_printoptions(threshold=np.nan)
        im = np.array(im, dtype=np.float32)
        H = im.shape[0]  # 高
        W = im.shape[1]  # 宽
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]

        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]

        # np.ndarray to torch.tensor
        # im = np.expand_dims(im,axis=2)
        # transformer = transforms.Compose([transforms.ToTensor()])
        # im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))  # 返回图像路径列表
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)  # Image.open()函数只是保持了图像被读取的状态，但是图像的真实数据并未被读取，
        # im=cv2.imread(fn)                   # 因此如果对需要操作图像每个元素，如输出某个像素的RGB值等，需要执行对象的load()方法读取数据。
        im = np.array(im, dtype=np.float32)
        # im = np.expand_dims(im, axis=2)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def ssim(prediction, target):  # 计算结构相似性
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


if __name__ == "__main__":

    # Validation Set
    AstroImage = opt.val_dirs

    valid_dict = {
        "AstroImage": validation_kodak(AstroImage),  # 读取这个验证集的图片  转化成矩阵列表
    }
    writer = SummaryWriter()
    # Noise adder
    noise_adder = AugmentNoise(style=opt.noisetype)  # 添加噪声

    # Network
    # network = ResUnetPlusPlus(opt.n_channel)#
    network = UNet(in_nc=opt.n_channel, out_nc=opt.n_channel, n_feature=opt.n_feature)
    # network = RIDNET(opt)
    # NAFNet
    # img_channel = opt.n_channel
    # width = 32
    # enc_blks = [2,2,2,20]
    # middle_blk_num = 2
    # dec_blks = [2,2,2,2]

    # network = NAFNet(img_channel=img_channel,width=width,middle_blk_num=middle_blk_num,
    #                  enc_blk_nums=enc_blks,dec_blk_nums=dec_blks)
    # network = SRMNet()
    # network = MIRNet_v2()
    # network = SCUNet()
    # network = CDLNet()
    # NAFNet
    # img_channel = opt.n_channel
    # width = 32
    # enc_blks = [2,2,2,20]
    # middle_blk_num = 2
    # dec_blks = [2,2,2,2]
    #
    # network = NAFNet(img_channel=img_channel,width=width,middle_blk_num=middle_blk_num,
    #                  enc_blk_nums=enc_blks,dec_blk_nums=dec_blks)
    # network = RIDNET(opt)
    # if opt.parallel:
    network = torch.nn.DataParallel(network)
    
    network = network

    checkpoint(network, 0, "model")  # 初始化权重
    print('init finish')

    network.load_state_dict(torch.load(opt.model_weight_path, map_location=device))

    network.eval()
    
    input_t = torch.randn(1, 1, 256, 256)
    
    traced_model = torch.jit.trace(network, input_t, check_trace=False)
    
    inputs = [torch_mlu.Input((1,1,256,256), dtype=torch.float, format=torch.contiguous_format)]
    
    compile_spec = {
        "inputs": inputs,
        "device": {"mlu_id": 0},
        "enabled_precisions": {torch.float},
    }
    
    compiled_model = torch_mlu.ts.compile(traced_model, **compile_spec)
    
    print("Ready Trace\n")
    
    # validation
    save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                   systime)  # ./results/unet_gauss25_tianwen1_b4e100/时间日期
    validation_path = os.path.join(save_model_path, "validation")
    os.makedirs(validation_path, exist_ok=True)
    np.random.seed(101)
    valid_repeat_times = {"AstroImage": 1}
    
    null_tensor_1 = torch.zeros([1,1,256,128]).to(device)
    null_tensor_1 = null_tensor_1.to(torch.uint8)
    null_tensor_2 = torch.zeros([1,1,128,256]).to(device)
    null_tensor_2 = null_tensor_2.to(torch.uint8)
    null_tensor_3 = torch.zeros([1,1,128,128]).to(device)
    null_tensor_3 = null_tensor_3.to(torch.uint8)
    print("null_tensor_1.dtype: ", null_tensor_1.dtype)

    print("\nReady For\n")
    T1 = time.time()
    for valid_name, valid_images in valid_dict.items():  # 1次循环
        print(valid_name)
        # print(valid_name)
        # print(valid_images)
        psnr_result = []
        ssim_result = []
        repeat_times = 1
        for k in range(repeat_times):  # 循环10 3 20次
            count=1
            for idx, im in enumerate(valid_images):
                # print("count: ", count)
                print(type(im))
                im = im.astype(np.uint8)
                print("\nim.dtype:", im.dtype)
                origin255 = im.copy()  # 原图
                origin255 = origin255.astype(np.uint8)

                im = np.array(im, dtype=np.uint8) / 255.0
                noisy_im = noise_adder.add_valid_noise(im)  # 加噪图

                noisy255 = noisy_im.copy()
                print("\nnoisy255.dtype:", noisy255.dtype)

                noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                   255).astype(np.uint8)

                if hasattr(torch.mlu, 'empty_cache'):
                    torch.mlu.empty_cache()

                ps = 256
                step = 128  ######128
                img = noisy_im
                W, H = img.shape[0: 2]
                # this part create a border around image, to eliminate any checkerboard pattern
                w_diff = (ps - (W % 256))
                w_c = ps + int(np.ceil(w_diff / 2))
                w_f = ps + int(np.floor(w_diff / 2))
                h_diff = (ps - (H % 256))
                h_c = ps + int(np.ceil(h_diff / 2))
                h_f = ps + int(np.floor(h_diff / 2))
                w = W + w_c + w_f
                h = H + h_c + h_f

                image = np.zeros((w, h, 1), dtype=np.uint8)
                image[w_c: w_c + W, h_c: h_c + H, 0] = img
                prediction = np.zeros((w, h), dtype=np.uint8)
                with torch.no_grad():
                    for i in range(0, w, step):
                        for j in range(0, h, step):
                            in_patch = image[i: i + ps, j: j + ps]  # (256,256)
                            # in_patch = np.expand_dims(in_patch, axis=0)
                            transformer = transforms.Compose([transforms.ToTensor()])
                            in_patch = transformer(in_patch)  # (1,256,256)
                            in_patch = torch.unsqueeze(in_patch, 0)  # (1,1,256,256)
                            # new_image = network(in_patch)  # (1,1,256,256)
                            # print("111")
                            # print(in_patch.shape)
                            
                            # UNNORMAL!
                            in_patch = in_patch.to(device)
                            if in_patch.shape == (1,1,256,128):
                                in_patch = torch.cat((in_patch, null_tensor_1), dim=3)
                                # in_patch = F.pad(in_patch, (0,128,0,0), mode='constant', value=0)
                                # print("unnormal new_shape after cat: ", in_patch.shape)
                                
                                new_image = compiled_model(in_patch.to('mlu'))
                                # print("unnormal output:", new_image.shape)
                                new_image = new_image.cpu().numpy()  # (1,1,256,256)
                                # print("new_image", type(new_image))
                                #print("prediction", type(prediction))
                                prediction[i: i + ps, j: j + ps] = new_image[0, 0, :, :128] + prediction[i: i + ps, j: j + ps]
                            
                            elif in_patch.shape == (1,1,128,256):
                                in_patch = torch.cat((in_patch, null_tensor_2), dim=2)
                                # in_patch = F.pad(in_patch, (0,0,0,128), mode='constant', value=0)
                                # print("unnormal new_shape after cat: ", in_patch.shape)
                                
                                new_image = compiled_model(in_patch.to('mlu'))
                                new_image = new_image.to(torch.uint8)
                                # print("unnormal output:", new_image.shape)
                                new_image = new_image.cpu().numpy()  # (1,1,256,256)
                                # prediction[i: i + ps, j: j + ps] += new_image[0, 0, :128, :]
                                prediction[i: i + ps, j: j + ps] = new_image[0, 0, :128, :] + prediction[i: i + ps, j: j + ps]
                                
                            elif in_patch.shape == (1,1,128,128):
                                in_patch = torch.cat((in_patch, null_tensor_3), dim=3) #(1,1,128,256)
                                in_patch = torch.cat((in_patch, null_tensor_2), dim=2)
                                
                                # in_patch = F.pad(in_patch, (0,128,0,128), mode='constant', value=0)
                                # print("unnormal new_shape after pad: ", in_patch.shape)
                                
                                new_image = compiled_model(in_patch.to('mlu'))
                                new_image = new_image.to(torch.uint8)
                                # print("unnormal output:", new_image.shape)
                                new_image = new_image.cpu().numpy()  # (1,1,256,256)
                                #print("new_image", type(new_image))
                                #print("prediction", type(prediction))
                                # prediction[i: i + ps, j: j + ps] += new_image[0, 0,:128,:128]
                                prediction[i: i + ps, j: j + ps] = new_image[0, 0,:128,:128] + prediction[i: i + ps, j: j + ps]
                                
                            else:
                                new_image = compiled_model(in_patch.to('mlu'))
                                new_image = new_image.to(torch.uint8)
                                # print("normal output:", new_image.shape)
                                
                                new_image = new_image.cpu().numpy()  # (1,1,256,256)
                                # prediction[i: i + ps, j: j + ps] += new_image[0, 0, :, :]
                                prediction[i: i + ps, j: j + ps] = new_image[0, 0, :, :] + prediction[i: i + ps, j: j + ps]
                                
                            # new_image = new_image.numpy()  # (1,1,256,256)
                            
                            # print("222\n")

                    prediction = (prediction / ((ps / step) ** 2))[w_c: w_c + W, h_c: h_c + H]

                pred255 = np.clip(prediction * 255.0 + 0.5, 0,
                                  255).astype(np.uint8)

                cur_psnr = calculate_psnr(origin255.astype(np.uint8),
                                          pred255.astype(np.uint8))
                psnr_result.append(cur_psnr)
                cur_ssim = calculate_ssim(origin255.astype(np.uint8),
                                          pred255.astype(np.uint8))
                ssim_result.append(cur_ssim)

                # visualization

                save_path = os.path.join(
                    validation_path,
                    "{}_{:03d}_clean.png".format(
                        valid_name, idx))
                Image.fromarray(origin255).convert('RGB').save(
                    save_path)  # 存储原图

                save_path = os.path.join(
                    validation_path,
                    "{}_{:03d}_noisy.png".format(
                        valid_name, idx))
                Image.fromarray(noisy255).convert('RGB').save(
                    save_path)  # 存储噪声图

                save_path = os.path.join(
                    validation_path,
                    "{}_{:03d}_denoised.png".format(
                        valid_name, idx))
                Image.fromarray(pred255).convert('RGB').save(save_path)  # 存储去噪图
                '''
                fromarray:array到image的转换，
                '''
            count = count+1    
        T2 = time.time()
        print("\nTIME USED:", T2-T1)
        psnr_result = np.array(psnr_result)
        avg_psnr = np.mean(psnr_result)
        avg_ssim = np.mean(ssim_result)
        print("PSNR: ",avg_psnr)
        print("SSIM: ", avg_ssim)
        
        log_path = os.path.join(validation_path,
                                "A_log_{}.csv".format(valid_name))
        with open(log_path, "a") as f:
            f.writelines("{},{}\n".format(avg_psnr, avg_ssim))  # 存储信噪比
