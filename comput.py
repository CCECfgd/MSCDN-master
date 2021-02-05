import cv2
from skimage.measure import compare_ssim
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import glob
def comput(path1,path2):
    print(path1,path2)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))
    SSIM = compare_ssim(img1, img2,multichannel=True)
    PSNR = peak_signal_noise_ratio(img1, img2)
    return SSIM,PSNR
if __name__ == "__main__":
    gt_List = glob.glob(r"dataset/testdataset/outdoor/gt/*")
    _list = glob.glob(r"test_result/outdoor/wave/*")
    SSIM = []
    PSNR = []

    for i in range(len(gt_List)):
        ssim,psnr = comput(gt_List[i],_list[i])
        SSIM.append(ssim)
        PSNR.append(psnr)
    s = np.array(SSIM)
    p = np.array(PSNR)
    print(np.mean(s),np.mean(p))