import numpy as np
import matplotlib.pyplot as plt

# ====================== Common Kernels ======================
kernels = {
    "Identity": np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]),
    
    "Edge_Laplacian": np.array([[ 0, -1,  0],
                                [-1,  4, -1],
                                [ 0, -1,  0]]),
    
    "Sobel_Horizontal": np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]),
    
    "Sobel_Vertical": np.array([[-1,-2,-1],
                                [ 0, 0, 0],
                                [ 1, 2, 1]]),
    
    "Sharpen": np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]]),
    
    "Gaussian_Blur": np.array([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]]) / 16.0,
    
    "Emboss": np.array([[-2, -1, 0],
                        [-1,  1, 1],
                        [ 0,  1, 2]])
}

# ====================== YOUR IMPLEMENTATIONS ======================

def convolve2d(image, kernel, stride=1, padding=0):
    """
    Perform 2D convolution (forward pass).
    
    Parameters:
        image: 2D numpy array (H, W) or 3D (H, W, C)
        kernel: 2D or 3D kernel
        stride, padding
    
    Returns: convolved output
    """
    # TODO: Implement this function
    # Hint: Start with grayscale (2D). Then extend to 3 channels.
    for dimension in image.shape:
        if dimension == 3:
            raise NotImplementedError("3-channel convolution not implemented yet.")
    
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    H, W = image.shape
    KH, KW = kernel.shape
    
    out_H = (H - KH) // stride + 1
    out_W = (W - KW) // stride + 1
    
    output = np.zeros((out_H, out_W))
    
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+KH, w_start:w_start+KW]
            output[i, j] = np.sum(patch * kernel)
    
    return output


def max_pool2d(image, pool_size=2, stride=2):
    """Implement 2D max pooling"""
    # TODO: Implement this
    
    H, W = image.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+pool_size, w_start:w_start+pool_size]
            output[i, j] = np.max(patch)
    return output


def avg_pool2d(image, pool_size=2, stride=2):
    """Implement 2D average pooling"""
    # TODO: Implement this
    H, W = image.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start+pool_size, w_start:w_start+pool_size]
            output[i, j] = np.mean(patch)
    return output
