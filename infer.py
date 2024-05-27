## 라이브러리 추가하기
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import cv2

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_dir", default="./dataset", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")


args = parser.parse_args()

## 트레이닝 파라메터 설정하기

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
result_dir = args.result_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("result dir: %s" % result_dir)


## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize expects tensor with 1 channel for grayscale
])


## 네트워크 생성하기
net = UNet().to(device)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## 네트워크 학습시키기
st_epoch = 0

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
net.to(device)


with torch.no_grad():
    net.eval()
    loss_arr = []

    files = os.listdir("./dataset/test")
    id = 0
    
    for file in files:
        # forward pass
        fname = os.path.join("./dataset/test/", file)
        print(fname)
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        print(img.shape)

        new_size = (128, 128)
        rsimg = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        disimg = rsimg.copy()

        rsimg = rsimg.astype(np.float32) / 255.0

        if rsimg.ndim == 2:
            rsimg = rsimg[:, :, np.newaxis]

        print(rsimg.shape)

        img_tran = transform(rsimg)
        img_tran = img_tran.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        img_tran = img_tran.to(device)  # Ensure tensor is on the same device as the model

        print(img_tran.shape)

        output = net(img_tran)
        output = fn_tonumpy(fn_class(output))
        output = output[0].squeeze()

        # plt.subplot(1,2,1)
        # plt.imshow(disimg, cmap="gray")
        # plt.title("org")

        # plt.subplot(1,2,2)
        # plt.imshow(output, cmap='gray')
        # plt.title("infer")
        # plt.show()

        plt.imsave(os.path.join(result_dir, 'input_%04d.png' % id), disimg, cmap='gray')
        plt.imsave(os.path.join(result_dir, 'output_%04d.png' % id), output, cmap='gray')

        id += 1

