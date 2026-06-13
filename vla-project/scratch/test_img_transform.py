import os
import sys
import numpy as np
from PIL import Image

# Add current path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "OmniVLA", "inference")))

from utils_policy import transform_images_PIL_mask, transform_images_PIL

# Create a random test image (uint8, range 0-255)
np.random.seed(42)
img_array = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
pil_img = Image.fromarray(img_array)

# Mask of ones
mask = np.ones((96, 96, 3), dtype=np.float32)

# Run transforms
tensor_no_mask = transform_images_PIL(pil_img)
tensor_with_mask = transform_images_PIL_mask(pil_img, mask)

print("Original image array min/max:", img_array.min(), img_array.max())
print("\ntransform_images_PIL (No mask):")
print("  Tensor shape:", tensor_no_mask.shape)
print("  Min:", tensor_no_mask.min().item())
print("  Max:", tensor_no_mask.max().item())
print("  Mean:", tensor_no_mask.mean().item())

print("\ntransform_images_PIL_mask:")
print("  Tensor shape:", tensor_with_mask.shape)
print("  Min:", tensor_with_mask.min().item())
print("  Max:", tensor_with_mask.max().item())
print("  Mean:", tensor_with_mask.mean().item())
