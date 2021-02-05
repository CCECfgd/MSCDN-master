import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(orig_images_path, hazy_images_path,label_images_path):
    train_list = []
    val_list = []

    # image_list_haze = glob.glob(hazy_images_path + "*")																	#路径匹配
    # for i3-SSIM-b1-n10-aodnet-室外合成 in range(len(image_list_haze)):
    # 	image_list_haze[i3-SSIM-b1-n10-aodnet-室外合成]=image_list_haze[i3-SSIM-b1-n10-aodnet-室外合成].replace("\\","/")
    image_list_haze_index = os.listdir(hazy_images_path)  # 文件名
    image_dataset = []
    for i in image_list_haze_index:  # 添加路径，并组合为元组
        image_dataset.append((orig_images_path + i, hazy_images_path + i,label_images_path + i))

    train_list = image_dataset#[:14350]
    

    return train_list

def populate_val_list(orig_images_path, hazy_images_path,label_images_path):
    train_list = []
    val_list = []

    # image_list_haze = glob.glob(hazy_images_path + "*")																	#路径匹配
    # for i3-SSIM-b1-n10-aodnet-室外合成 in range(len(image_list_haze)):
    # 	image_list_haze[i3-SSIM-b1-n10-aodnet-室外合成]=image_list_haze[i3-SSIM-b1-n10-aodnet-室外合成].replace("\\","/")
    image_list_haze_index = os.listdir(hazy_images_path)  # 文件名
    image_dataset = []
    for i in image_list_haze_index:  # 添加路径，并组合为元组
        image_dataset.append((orig_images_path + i, hazy_images_path + i,label_images_path + i))

    #train_list = image_dataset[:12000]
    val_list = image_dataset

    return val_list


class dehazing_loader(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path,label_images_path, mode='train'):

        self.train_list = populate_train_list(orig_images_path, hazy_images_path,label_images_path)
        self.val_list = populate_train_list(orig_images_path, hazy_images_path,label_images_path)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_clean_path, data_hazy_path,data_depth_path = self.data_list[index]
        #print(data_clean_path)
        #data_clean = cv2.imread(data_clean_path)
        #data_hazy = cv2.imread(data_hazy_path)
        #data_depth = cv2.imread(data_depth_path)
        data_clean = Image.open(data_clean_path)
        data_hazy = Image.open(data_hazy_path)
        data_depth = Image.open(data_depth_path)
        #data_clean = cv2.resize(data_clean,(640,480),interpolation=cv2.INTER_CUBIC)
        #data_hazy = cv2.resize(data_hazy,(640,480),interpolation=cv2.INTER_CUBIC)
        #data_depth = cv2.resize(data_depth,(640,480),interpolation=cv2.INTER_CUBIC)
        # data_depth = cv2.merge([data_depth,data_depth,data_depth])


        data_clean = data_clean.resize((640, 480), Image.ANTIALIAS)
        data_hazy = data_hazy.resize((640, 480), Image.ANTIALIAS)
        data_depth = data_depth.resize((640, 480), Image.ANTIALIAS)

        #data_depth = Image.merge('RGB',[data_depth,data_depth,data_depth])

        data_clean = (np.asarray(data_clean) / 255.0)
        data_hazy = (np.asarray(data_hazy) / 255.0)
        data_depth = (np.asarray(data_depth) / 255.0)

        # data_clean = data_clean/255.0
        # data_hazy = data_hazy/255.0
        # data_depth = data_depth/255.0

        data_clean = torch.from_numpy(data_clean).float()
        data_hazy = torch.from_numpy(data_hazy).float()
        data_depth = torch.from_numpy(data_depth).float()

        return data_clean.permute(2, 0, 1), data_hazy.permute(2, 0, 1) ,data_depth.unsqueeze(0)

    def __len__(self):
        return len(self.data_list)

