import numpy as np


def _cc2d_2layer(new_img: np.array, kernel: np.array, padded_img: np.array):
    def _pp_kernel_sum(_ih, _iw):
        return np.multiply(kernel, padded_img[_ih: _ih + kernel.shape[0], _iw: _iw + kernel.shape[1]]).sum()

    for j in range(new_img.shape[0]):
        for i in range(new_img.shape[1]):
            new_img[j, i] = _pp_kernel_sum(j, i)

    return new_img


def cross_correlation_2d(img: np.array, kernel: np.array):
    """Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """

    kernel_m_size, kernel_n_size = kernel.shape[:2]
    k_m, k_n = (kernel_m_size - 1) // 2, (kernel_n_size - 1) // 2
    img_channels = 1 if img.ndim != 3 else 3
    if img_channels == 1:
        padded_image = np.pad(img, ((k_m, k_m), (k_n, k_n)), 'constant')
    else:
        padded_image = np.pad(img, ((k_m, k_m), (k_n, k_n), (0, 0)), 'constant')

    output_img = np.zeros(img.shape)

    if img_channels == 1:
        output_img = _cc2d_2layer(output_img, kernel, padded_image)
        print(output_img)
        return output_img
    else:
        for i in range(img_channels):
            output_img[:, :, i] = _cc2d_2layer(output_img[:, :, i], kernel, padded_image[:, :, i])
        print(output_img)
        return output_img


def convolve_2d(img: np.array, kernel: np.array):
    """Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    kernel = np.flipud(np.fliplr(kernel))
    return cross_correlation_2d(img, kernel)


def gaussian_blur_kernel_2d(sigma: float, height: int, width: int):
    """Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    """
    k_y = (height - 1) // 2
    k_x = (width - 1) // 2
    x, y = np.meshgrid(np.linspace(-k_x, k_x, num=width), np.linspace(-k_y, k_y, num=height))

    x2py2 = np.sqrt(x ** 2 + y ** 2)
    pre = 1 / (2.0 * np.pi * sigma ** 2)

    gauss = np.exp(-0.5 * (x2py2 ** 2 / sigma ** 2)) * pre

    return gauss / np.sum(gauss)


def low_pass(img: np.array, sigma: float, size):
    """Filter the image as if it's filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter suppresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    """Filter the image as if it's filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    """
    # reminder: F + a(F - F * H)
    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio, scale_factor):
    """This function adds two images to create a hybrid image, based on
    parameters specified by the user."""
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
