import torch
import torchvision
import torch.optim
#import att_model as model
import wave_u_net as model
import numpy as np
import U_net as model2
import torch.nn as nn
from PIL import Image
import glob
import time,os
from U_net import Unet

def dehaze_image( image_depth_path,image_hazy_path,Id,spath):
    """
    当前，输入为有雾图像和无雾图像，有雾图像和无雾图像都转为灰度图进人网络
    :param image_down_path:深度图下采样，当前用有雾图像的灰度图代替
    :param image_label_path:深度图，用无雾图像灰度图代替
    :param image_add_path:有雾彩色图像
    :param Id:
    :return:
    """
    print(image_hazy_path,image_depth_path)
    img_hazy = Image.open(image_hazy_path)
    img_depth = Image.open(image_depth_path)
    #img_gt = Image.open(image_gt_path)

    img_hazy = img_hazy.resize((640,480), Image.ANTIALIAS)
    #img_gt = img_gt.resize((640,480), Image.ANTIALIAS)
    img_depth = img_depth.resize((640,480), Image.ANTIALIAS)
    #img_depth = Image.merge('RGB',[img_depth,img_depth,img_depth])

    img_hazy = (np.asarray(img_hazy) / 255.0)
    #img_gt = (np.asarray(img_gt) / 255.0)
    img_depth = (np.asarray(img_depth) / 255.0)

    img_hazy = torch.from_numpy(img_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)
    #img_gt = torch.from_numpy(img_gt).float().permute(2, 0, 1).cuda().unsqueeze(0)
    #img_depth = torch.from_numpy(img_depth).float().permute(2, 0, 1).cuda().unsqueeze(0)
    img_depth = torch.from_numpy(img_depth).float().cuda().unsqueeze(0).unsqueeze(0)
    #dehaze_net = model.MSFAN()

    #dehaze_net = Unet().cuda()
    dehaze_net = model.LAST_U_net()
    dehaze_net = nn.DataParallel(dehaze_net).cuda()
    #dehaze_net = model.MSFAN().cuda()
    dehaze_net.load_state_dict(torch.load('/home/amax/share/FGD/U-Net/trained_model/outdoor/wave-U-net/Epoch8.pth'))
    #dehaze_net.load_state_dict(torch.load('trained_model/outdoor/new_model/MSDFN.pth'))

    clean_image = dehaze_net(img_depth,img_hazy)
    #clean_image = dehaze_net(img_hazy,img_depth)
    temp_tensor = (clean_image,0)
    index = image_depth_path.split('/')[-1]
    if not os.path.exists("test_result/outdoor/%s" % spath):
        os.mkdir("test_result/outdoor/%s" % spath)
    #torchvision.utils.save_image(torch.cat((img_hazy,clean_image),0), "test_result/real/%s/%s" % (s,index))


    torchvision.utils.save_image(clean_image, "test_result/outdoor/%s/%s" % (spath,index))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    spath = 'outdoor'
    depth_list = glob.glob(r"dataset/testdataset/%s/depth/*" % spath)
    hazy_list = glob.glob(r"dataset/testdataset/%s/hazy/*" % spath)
    #gt_list = glob.glob(r"/home/amax/share/FGD/IRDN-master/dataset/testdataset/outdoor/gt/*")
    s = time.time()
    for Id in range(len(depth_list)):
        dehaze_image(depth_list[Id],hazy_list[Id],Id,spath)
        print(depth_list[Id], "done!")
    e = time.time()
    print((e-s)/492)
