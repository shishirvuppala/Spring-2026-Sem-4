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
    pass


def max_pool2d(image, pool_size=2, stride=2):
    """Implement 2D max pooling"""
    # TODO: Implement this
    pass


def avg_pool2d(image, pool_size=2, stride=2):
    """Implement 2D average pooling"""
    # TODO: Implement this
    pass
