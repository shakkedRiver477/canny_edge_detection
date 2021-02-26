import matplotlib.pyplot as plt
import numpy as np
import non_maximum_suppuration_file
from imageio import imread , imsave
from scipy import signal
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
from skimage import color


def read_image(filename, representation=2):
    """
    reads a image from a file if representation=2 returns a rgb image if 1 returns gray scale image
    :param filename: the image file path
    :param representation: 1 for gray scale image and 2 for rgb default is rgb
    :return: a ndArray representing the image
    """
    img = imread(filename)
    if representation == 1 and img.ndim == 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    if img.dtype != np.float64:
        img = img.astype(np.float64)
    return img


def get_filter_vec(n):
    """
    genrate vector filter the size of n a positive int(wont enforce) which would ne a line in pascal triangle
    :param n: the size we want the filter to be
    :type n:  positive int
    :return: the filter we want
    :rtype: np.float64 ndArray of shape(1,n)
    """
    if (n == 1):
        return np.array([[1]])
    kernal = np.array([[1, 1]])
    if (n == 2):
        return kernal
    array_to_return = np.array([[1, 1]])
    for i in range(n - 2):
        array_to_return = convolve2d(array_to_return, kernal)
    return (array_to_return / (2 ** (n - 1))).astype(np.float64)


def blur(im, filter_vec):
    """
    gets a image and blur it by convolving with a vector filter. works directly on im
    :param im: the image
    :type im: ndArray of np.float64 shape (m,n)
    :param filter_vec: ndArray of np.float64 shape (1,k)
    :type filter_vec:
    :return:im after blur
    :rtype:a copy of the image
    """
    image = convolve(im, filter_vec)
    image = convolve(image, filter_vec.T)
    return image


def conv_der(im):
    """
    computs the abs of derivative approximation of un image with convolution and the angel of the gradient
    :param im: the image
    :type im: ndArray of type np.float64
    :return: the abs of the derivative of the image
    :rtype: tuple ndArray of type np.float64 representing the abs of the gradient
    :and and ndArray of type np.float64 representing the angle of change between pi and -pi
    """
    dims = im.shape
    if len(dims) == 3:
        im = im[:, :, 0]

    dx = signal.convolve2d(im, np.array([[0.5, 0, -0.5]]), mode="same")
    dy = signal.convolve2d(im, np.array([[0.5, 0, -0.5]]).T, mode="same")
    angels = np.arctan(dy / dx)
    angels = np.nan_to_num(angels, nan=np.pi / 2)
    if len(dims) == 3:
        return np.array([np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)]).clip(0, 1), angels
    else:
        return (np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2) * 2).clip(0, 1), angels


def double_threshold(im, t1, t2):
    """
    return a tuple of two binary images the first correspond to where the image is higher then t1 and the second is the
    same but for t2
    :param im: the image
    :param t1: the lower threshold
    :param t2: the upper threshold
    :return: return a tuple of two binary images the firs correspond to where the image is higher then t1 and the second
    is the same but for t2
    """
    im1 = np.zeros_like(im, dtype=bool)
    im1[im >= t1] = True
    im2 = np.zeros_like(im, dtype=bool)
    im2[im >= t2] = True
    return im1, im2


def supers_weak_edges(im1, im2):
    """
    keep the week edges im1 only if they are near a strong edge im2
    :param im1: a binary image of the week edges
    :param im2: a binary image of the strong edges
    :return: im2 + all week edges that are near
    """
    dims = im1.shape
    canidets = im1.astype(np.float64) - im2.astype(np.float64)
    pointes = np.where(canidets > 0)
    re = im2.copy()
    for i in range(len(pointes[0])):
        y, x = pointes[0][i], pointes[1][i]
        inveorment = [im2[max(y - 1, 0), x],
                      im2[max(y - 1, 0), min(x + 1, dims[1] - 1)],
                      im2[y, min(x + 1, dims[1] - 1)],
                      im2[min(y + 1, dims[0] - 1), min(x + 1, dims[1] - 1)],
                      im2[min(y + 1, dims[0] - 1), x],
                      im2[min(y + 1, dims[0] - 1), max(x - 1, 0)],
                      im2[y, max(x - 1, 0)],
                      im2[max(y - 1, 0), max(x - 1, 0)]]
        if True in inveorment:
            re[y, x] = True
    return re


def non_maximum_suppuration(grad, angels):
    """

    :param result:
    :param angels:
    :return:
    """
    result = grad.copy()
    angels = angels * 8 / np.pi
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if 3 < abs(angels[i, j]) <= 4:
                if not (grad[i, j] > grad[i + 1, j] and (grad[i, j] > grad[i - 1, j])):
                    result[i, j] = 0
            elif 1 < angels[i, j] <= 3:
                if not (grad[i, j] > grad[i - 1, j + 1] and (grad[i, j] > grad[i + 1, j - 1])):
                    result[i, j] = 0
            elif -1 < angels[i, j] <= 1:
                if not (grad[i, j] > grad[i, j + 1] and (grad[i, j] > grad[i, j - 1])):
                    result[i, j] = 0
            else:
                if not (grad[i, j] > grad[i + 1, j + 1] and (grad[i, j] > grad[i - 1, j - 1])):
                    result[i, j] = 0

    return result


def edge_detection(image, t1, t2):
    """
    gets a gray scale image and apply canny edge detection om it
    :param image: the image
    :param t1:the lower threshold pixels that are above this threshold would be 1 iff they are above t2 or one of their
     neighbors is.
    :param t2:the upper threshold
    :return: a binary image such that only edges are white
    """
    image = blur(image, get_filter_vec(5))
    grad, angels = conv_der(image)
    grad = non_maximum_suppuration_file.non_maximum_suppuration(grad, angels)
    im1, im2 = double_threshold(grad, t1, t2)
    result = supers_weak_edges(im1, im2).astype(np.float64)
    return result


if __name__ == '__main__':
    im = read_image("tools.jpg", 1)
    plt.imshow(im, cmap="gray")
    plt.show()
    im = edge_detection(im, 0.0, 0.1)
    plt.imshow(im, cmap="gray")
    plt.show()
    imsave("ilay.jpeg",im)
