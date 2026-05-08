import numpy as np
import matplotlib.pyplot as plt

from cnn import convolve2d, max_pool2d, avg_pool2d, kernels

# ====================== Helper Functions ======================

def create_synthetic_images(size=128):
    """Create clean synthetic images perfect for kernel visualization"""
    images = {}
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # 1. Circle / Disk
    images["Circle"] = (X**2 + Y**2 < 0.5).astype(np.float32)
    
    # 2. Horizontal Gradient
    images["Horizontal_Gradient"] = (X + 1) / 2
    
    # 3. Vertical Gradient
    images["Vertical_Gradient"] = (Y + 1) / 2
    
    # 4. Checkerboard
    images["Checkerboard"] = ((np.floor(X*4) + np.floor(Y*4)) % 2)
    
    # 5. Radial Gradient (Gaussian-like blob)
    images["Radial_Blob"] = np.exp(- (X**2 + Y**2) / 0.3)
    
    # 6. Diagonal Edge
    images["Diagonal_Edge"] = (X + Y > 0).astype(np.float32)
    
    # 7. Sine Wave Pattern (good for frequency understanding)
    images["Sine_Waves"] = (np.sin(8 * X) + np.sin(8 * Y)) / 2 + 0.5
    
    return images

# ====================== Choose Test Image ======================

# Generate images
synthetic_images = create_synthetic_images(size=128)
image_list = list(synthetic_images.keys())

test_image_name = input(f"Enter the name of the test image ({', '.join(image_list)}): ")

try :
    test_image = synthetic_images[test_image_name]
except KeyError:
    print(f"Invalid image name. Defaulting to 'Circle'.")
    test_image_name = "Circle"
    test_image = synthetic_images[test_image_name]

print(f"\nSelected image for experiments: **{test_image_name}**")
plt.figure(figsize=(5,5))
plt.imshow(test_image, cmap='gray')
plt.title(f"Selected Test Image: {test_image_name}")
plt.axis('off')
plt.show()

# ====================== EXPERIMENT 1: Effect of Different Kernels ======================
print("\n" + "="*70)
print("EXPERIMENT 1: Applying Different Kernels")
print("="*70)

for name, kernel in kernels.items():
    print(f"\n→ Applying kernel: {name}")
    
    conv_output = convolve2d(test_image, kernel, stride=1, padding=1)
    relu_output = np.maximum(conv_output, 0)
    pooled_output = max_pool2d(relu_output, pool_size=2, stride=2)
    
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(test_image, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(conv_output, cmap='gray')
    plt.title(f"Convolution\n({name})")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(relu_output, cmap='gray')
    plt.title("After ReLU")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(pooled_output, cmap='gray')
    plt.title("After MaxPool 2x2")
    plt.axis('off')
    
    plt.suptitle(f"Kernel: {name} | Image: {test_image_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

# ====================== 4. EXPERIMENT 2: Effect of Stride and Padding ======================
print("\n" + "="*70)
print("EXPERIMENT 2: Effect of Stride and Padding (using Laplacian kernel)")
print("="*70)

kernel = kernels["Edge_Laplacian"]

configs = [
    ("No Padding, Stride=1", 0, 1),
    ("Padding=1, Stride=1", 1, 1),
    ("Padding=1, Stride=2", 1, 2),
    ("No Padding, Stride=2", 0, 2),
]

plt.figure(figsize=(15, 8))
for i, (title, pad, stride) in enumerate(configs):
    conv_out = convolve2d(test_image, kernel, stride=stride, padding=pad)
    plt.subplot(2, 2, i+1)
    plt.imshow(conv_out, cmap='gray')
    plt.title(title + f"\nShape: {conv_out.shape}")
    plt.axis('off')
plt.suptitle(f"Effect of Padding & Stride | Image: {test_image_name}", fontsize=16)
plt.tight_layout()
plt.show()

# ====================== 5. EXPERIMENT 3: Max vs Average Pooling ======================
print("\n" + "="*70)
print("EXPERIMENT 3: Max Pooling vs Average Pooling")
print("="*70)

conv_out = convolve2d(test_image, kernels["Edge_Laplacian"], stride=1, padding=1)
relu_out = np.maximum(conv_out, 0)

max_p = max_pool2d(relu_out, pool_size=2, stride=2)
avg_p = avg_pool2d(relu_out, pool_size=2, stride=2)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(relu_out, cmap='gray')
plt.title("After Conv + ReLU")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(max_p, cmap='gray')
plt.title("Max Pooling")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(avg_p, cmap='gray')
plt.title("Average Pooling")
plt.axis('off')

plt.suptitle(f"Max vs Avg Pooling | Image: {test_image_name}", fontsize=16)
plt.tight_layout()
plt.show()