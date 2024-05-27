

import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from PIL import Image


def img_to_numpy(img_path):
    # 이미지를 열고 numpy 배열로 변환합니다.
    with Image.open(img_path) as img:
        img_array = np.array(img)
    return img_array

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        files = os.listdir(self.data_dir)

        # .jpg 파일 중에서 'label'로 시작하는 파일을 필터링합니다.
        lst_input = [file for file in files if not file.startswith('label')]

        lst_input.sort()
        self.lst_input = lst_input

    def img_to_numpy(self,img_path):
        # 이미지를 열고 numpy 배열로 변환합니다.
        with Image.open(img_path) as img:
            img_array = np.array(img)
        return img_array

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):

        # print("list input index = ", self.lst_input[index])
        fname = os.path.join(self.data_dir, self.lst_input[index])
        # print(fname)        
        input = self.img_to_numpy(fname)
        
        fname = os.path.join(self.data_dir, "label_" + self.lst_input[index])
        # print(fname)        
        label = self.img_to_numpy(fname)

        # print("input shape ", input.shape, input.ndim)
        # print("label shape ", label.shape, label.ndim)


        label = label/255.0
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # print("input shape ", input.shape, input.ndim)    
        # print("label shape ", label.shape, label.ndim)    
        
        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data



## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data



if __name__ == "__main__":

    data_dir = "./dataset/train"
    udset = Dataset(data_dir)
    dload = torch.utils.data.DataLoader(udset, batch_size=4, shuffle=False)
    fs = next(iter(dload))
    print(fs['input'])

    
    
    






