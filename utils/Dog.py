import numpy as np
import cv2

def genGaussianKernel(ksize, sigma):
    half_ksize = ksize // 2
    C = 2 * np.pi * sigma * sigma
    x = y = np.linspace(-half_ksize, half_ksize, ksize)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / C
    return kernel

def zerosCrossing(src, thresh):
    dsize = (src.shape[1], src.shape[0])
    M = np.array([[1, 0, -1], [0, 1, 0]], dtype=np.float32)
    shift_left = cv2.warpAffine(src, M, dsize)
    M = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    shift_right = cv2.warpAffine(src, M, dsize)

    M = np.array([[1, 0,  0], [0, 1, -1]], dtype=np.float32)
    shift_up = cv2.warpAffine(src, M, dsize)
    M = np.array([[1, 0,  0], [0, 1, 1]], dtype=np.float32)
    shift_down = cv2.warpAffine(src, M, dsize)

    M = np.array([[1, 0,  1], [0, 1, 1]], dtype=np.float32)
    shift_right_down = cv2.warpAffine(src, M, dsize)
    M = np.array([[1, 0,  -1], [0, 1, -1]], dtype=np.float32)
    shift_left_up = cv2.warpAffine(src, M, dsize)

    M = np.array([[1, 0,  1], [0, 1, -1]], dtype=np.float32)
    shift_right_up = cv2.warpAffine(src, M, dsize)
    M = np.array([[1, 0,  -1], [0, 1, 1]], dtype=np.float32)
    shift_left_down = cv2.warpAffine(src, M, dsize)

    shift_left_right_sign = (shift_left * shift_right)
    shift_up_down_sign = (shift_up * shift_down)
    shift_rd_lu_sign = (shift_right_down * shift_left_up)
    shift_ru_ld_sign = (shift_right_up * shift_left_down)

    shift_left_right_norm = abs(shift_left - shift_right)
    shift_up_down_norm = abs(shift_up - shift_down)
    shift_rd_lu_norm = abs(shift_right_down - shift_left_up)
    shift_ru_ld_norm = abs(shift_right_up - shift_left_down)

    candidate_zero_crossing = \
        ((shift_left_right_sign < 0) & (shift_left_right_norm > thresh)).astype('uint8') +\
        ((shift_up_down_sign < 0) & (shift_up_down_norm > thresh)).astype('uint8') + \
        ((shift_rd_lu_sign < 0) & (shift_rd_lu_norm > thresh)).astype('uint8') + \
        ((shift_ru_ld_sign < 0) & (shift_ru_ld_norm > thresh)).astype('uint8')

    ResImg = np.zeros(shape=src.shape, dtype=np.uint8)
    ResImg[candidate_zero_crossing >= 2] = 255

    return ResImg


def DoG(src, ksize, sigma, k, thresh=None, alpha=0.01):
    sigma2 = sigma / k
    kernel_1 = genGaussianKernel(ksize=ksize, sigma=sigma)
    kernel_2 = genGaussianKernel(ksize=ksize, sigma=sigma2)
    kernel = kernel_1 - kernel_2

    DoG_img = cv2.filter2D(src=src, ddepth=cv2.CV_32FC1, kernel=kernel)
    if thresh is None:
        thresh = abs(DoG_img).max() * alpha
    edge_image = zerosCrossing(src=DoG_img, thresh=thresh)
    return edge_image