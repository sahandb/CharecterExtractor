# imports
import cv2
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import scipy.ndimage


def colorPx(img1, img2, fimg):
    # threshold
    thresholdedImg1 = (img1 >= 1) * img1
    thresholdedImg2 = (img2 >= 1) * img2

    xx = list()
    yy = list()
    for i in range(thresholdedImg1.shape[0]):
        for j in range(thresholdedImg1.shape[1]):
            if thresholdedImg1[i, j] == 1 or thresholdedImg2[i, j] == 1:
                xx.append(i)
                yy.append(j)

    for pixelX in range(len(xx)):
        thresholdedImg1[xx[pixelX] - 8:xx[pixelX] + 8, yy[pixelX] - 8:yy[pixelX] + 8] = 1

    xorImg = np.logical_xor(fimg, thresholdedImg1).astype(np.float32)
    newImg = np.zeros((fimg.shape[0], fimg.shape[1], 3), dtype=np.float32)
    newImg[:, :, 0] = xorImg
    newImg[:, :, 1] = fimg
    newImg[:, :, 2] = fimg
    return newImg


def window_sum(image, win_shape):
    win_sum = np.cumsum(image, axis=0)
    win_sum = (win_sum[win_shape[0]:-1] - win_sum[:-win_shape[0] - 1])

    win_sum = np.cumsum(win_sum, axis=1)
    win_sum = (win_sum[:, win_shape[1]:-1] - win_sum[:, :-win_shape[1] - 1])

    return win_sum


def extractPatternFrequency(image, template):
    image_shape = image.shape

    image = np.array(image, dtype=np.float64, copy=False)

    pad_width = tuple((width, width) for width in template.shape)
    image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)

    image_window_sum = window_sum(image, template.shape)
    image_window_sum2 = window_sum(image ** 2, template.shape)

    template_mean = template.mean()
    template_volume = np.prod(template.shape)
    template_ssd = np.sum((template - template_mean) ** 2)

    # correlation
    xcorr = fftconvolve(image, template[::-1, ::-1], mode="valid")[1:-1, 1:-1]

    numerator = xcorr - image_window_sum * template_mean
    denominator = image_window_sum2

    image_window_sum = np.multiply(image_window_sum, image_window_sum)
    image_window_sum = np.divide(image_window_sum, template_volume)

    denominator -= image_window_sum
    denominator *= template_ssd
    denominator = np.maximum(denominator, 0)  # sqrt of negative number not allowed
    denominator = np.sqrt(denominator)

    response = np.zeros_like(xcorr, dtype=np.float64)

    # avoid zero-division
    mask = denominator > np.finfo(np.float64).eps
    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        d0 = (template.shape[i] - 1) // 2
        d1 = d0 + image_shape[i]
        slices.append(slice(d0, d1))

    return response[tuple(slices)]


def extractPatternSpatial(image,template):
    image_shape = image.shape

    image = np.array(image, dtype=np.float64, copy=False)

    pad_width = tuple((width, width) for width in template.shape)
    image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)

    image_window_sum = window_sum(image, template.shape)
    image_window_sum2 = window_sum(image ** 2, template.shape)

    template_mean = template.mean()
    template_volume = np.prod(template.shape)
    template_ssd = np.sum((template - template_mean) ** 2)
    # correlation
    xcorr = scipy.signal.correlate2d(image, template[::-1, ::-1], mode="valid")[1:-1, 1:-1]
    numerator = xcorr - image_window_sum * template_mean
    denominator = image_window_sum2

    image_window_sum = np.multiply(image_window_sum, image_window_sum)
    image_window_sum = np.divide(image_window_sum, template_volume)

    denominator -= image_window_sum
    denominator *= template_ssd
    denominator = np.maximum(denominator, 0)  # sqrt of negative number not allowed
    denominator = np.sqrt(denominator)

    response = np.zeros_like(xcorr, dtype=np.float64)

    # avoid zero-division
    mask = denominator > np.finfo(np.float64).eps
    response[mask] = numerator[mask] / denominator[mask]

    slices = []
    for i in range(template.ndim):
        d0 = (template.shape[i] - 1) // 2
        d1 = d0 + image_shape[i]
        slices.append(slice(d0, d1))

    return response[tuple(slices)]


# read image
fullImage = cv2.imread('./text.png', 0).astype(np.float32)
a = fullImage[42:54, 199:208]
a_3rt90 = fullImage[246:256, 271:282]
# frequency domain
aFrequencyBack = np.round(extractPatternFrequency(fullImage, a), 3)
aFrequencyRotBack = np.round(extractPatternFrequency(fullImage, a_3rt90), 3)
resultFrequency = colorPx(aFrequencyBack, aFrequencyRotBack, fullImage)

cv2.imshow("result Frequency domain", resultFrequency)

aSpatialBack = np.round(extractPatternSpatial(fullImage, np.rot90(a,2)), 3)
aSpatialRotBack = np.round(extractPatternSpatial(fullImage, np.rot90(a_3rt90,2)), 3)

resultSpatial = colorPx(aSpatialBack, aSpatialRotBack, fullImage)

cv2.imshow("Spatial Spatial domain", resultSpatial)

cv2.waitKey(0)