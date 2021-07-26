import numpy as np
import cv2
import math
#MPC
def MPCX(r,img):
    size = img.shape
    sum = size[0]*(size[1]-r)
    count = 0
    for i in range(len(img)):
        for j in range(len(img)-r):
            if np.sum(img[i,j:j+r+1]) == 0:
                count += 1

    return count/sum
def MPCY(r,img):
    size = img.shape
    sum = size[0]*(size[1]-r)
    count = 0
    for i in range(len(img)):
        for j in range(len(img)-r):
            if np.sum(img[j:j+r+1,i]) == 0:
                count += 1
    return count/sum
def MPCD(r,img):
    size = img.shape
    sum = (size[0]-r) * (size[1] - r)
    count = 0
    sum_mask = 0
    for i in range(len(img)-r):
        for j in range(len(img)-r):
            data = np.zeros(r+1)
            for k in range(r+1):
                data[k] = img[i+k,j+k]

            if np.sum(data) == 0:
                count +=1
                sum_mask +=1
            else: sum_mask +=1

    return count/sum
#ssim
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
#psnr
def psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))