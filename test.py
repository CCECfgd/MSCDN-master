import torch
import torchvision
import torch.optim
import wave_u_net as model
import numpy as np
import torch.nn as nn
from PIL import Image
import glob
import time,os


def dehaze_image( image_depth_path,image_hazy_path,Id,spath,pth_path):

    print(image_hazy_path,image_depth_path)
    img_hazy = Image.open(image_hazy_path)
    img_depth = Image.open(image_depth_path)

    img_hazy = img_hazy.resize((640,480), Image.ANTIALIAS)

    img_depth = img_depth.resize((640,480), Image.ANTIALIAS)


    img_hazy = (np.asarray(img_hazy) / 255.0)

    img_depth = (np.asarray(img_depth) / 255.0)

    img_hazy = torch.from_numpy(img_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)
    img_depth = torch.from_numpy(img_depth).float().cuda().unsqueeze(0).unsqueeze(0)
   
    dehaze_net = model.LAST_U_net()
    dehaze_net = nn.DataParallel(dehaze_net).cuda()

    dehaze_net.load_state_dict(torch.load(pth_path))
    
    clean_image = dehaze_net(img_depth,img_hazy)

    temp_tensor = (clean_image,0)
    index = image_depth_path.split('/')[-1]
    
    #torchvision.utils.save_image(torch.cat((img_hazy,clean_image),0), "test_result/real/%s/%s" % (s,index))

    torchvision.utils.save_image(clean_image, "test_result/outdoor/%s/%s" % (spath,index))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    spath = 'outdoor'
    pth_path = "'/home/amax/share/FGD/U-Net/trained_model/outdoor/MCDN.pth'"
    depth_list = glob.glob(r"dataset/test set/%s/depth/*" % spath)
    hazy_list = glob.glob(r"dataset/test set/%s/hazy/*" % spath)

    for Id in range(len(depth_list)):
        dehaze_image(depth_list[Id],hazy_list[Id],Id,spath,pth_pth)
        print(depth_list[Id], "done!")
   
