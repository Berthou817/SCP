import cv2
import otsu
import matplotlib.pyplot as plt
import numpy as np


def show_thresholds(src_img, dst_img, thresholds):
    """Visualise thresholds."""
    colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 128, 0), (0, 204, 102),
               (51, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 127)]
    # colors = [(6, 0, 0), (6, 4, 0), (6, 6, 0), (0, 4, 0), (0, 5, 2), (1, 6, 6), (0, 4, 6), (0, 0, 6), (4, 0, 6),
    #           (6, 0, 6), (6, 0, 3)]


    masks = otsu.multithreshold(src_img, thresholds)
    for i, mask in enumerate(masks):
        # for debugging masks
        # background = np.zeros((dst_img.shape[0], dst_img.shape[1], 3))
        # background[mask] = (255, 255, 255)
        # plt.figure()
        # plt.imshow(background)
        dst_img[mask] = colors[i]
    return dst_img

def Non_local_mean_filter(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #noise_image = double2uint8(image + np.random.randn(*image.shape) * 20)
    out_image = cv2.fastNlMeansDenoisingColored(img, h=1,hColor=10)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
    return out_image
def full_method(img, L=256, M=32):
    """Obtain thresholds and image masks directly from image.

    Abstracts away the need to handle histogram generation.
    """
    # Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    thresholds = otsu.modified_TSMO(hist, M=M, L=L)
    masks = otsu.multithreshold(img, thresholds)
    return thresholds, masks
def segmentation(img):
    L = 256  # number of levels
    M = 32  # number of bins for bin-grouping normalisation

    N = L // M
    # Load original image
    # img = cv2.imread(path,0)  # read image in as grayscale
    # img = Non_local_mean_filter(img)
    #img = cv2.fastNlMeansDenoising(img, h=5,templateWindowSize=7,searchWindowSize=7)
    # Blur image to denoise
    #img = cv2.GaussianBlur(img, (5, 5), 0)

    # Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    ### Do modified TSMO step by step
    # Normalise bin histogram
    norm_hist = otsu.normalised_histogram_binning(hist, M=M, L=L)

    # Estimate valley regions
    valleys = otsu.find_valleys(norm_hist)


    thresholds = otsu.threshold_valley_regions(hist, valleys, N)
    ###

    # modified_TSMO does all the steps above
    thresholds2 = otsu.modified_TSMO(hist, M=M, L=L)

    # Threshold obtained through default otsu method.
    otsu_threshold, _ = otsu.otsu_method(hist)

    # print('Otsu threshold: {}\nStep-by-step MTSMO: {}\nMTSMO: {}'.format(
    #     otsu_threshold, thresholds, thresholds2))


    # Illustrate thresholds
    img_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_3 = img_1.copy()
    img_auto = img_1.copy()

    show_thresholds(img, img_1, [thresholds[0]])
    show_thresholds(img, img_3, thresholds[0:3])
    show_thresholds(img, img_auto, thresholds)


    # return img_auto,img_3,img_1
    return img_1