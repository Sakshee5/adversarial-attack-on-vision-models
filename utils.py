from __future__ import annotations

import argparse
from math import log10
from typing import Tuple
import os

import numpy as np
import numpy.typing as npt
import cv2
from PIL import Image, ImageDraw, ImageFont

ImageF32 = npt.NDArray[np.float32]
VecF32 = npt.NDArray[np.float32]


def srgb2lin(x: ImageF32) -> ImageF32:
    """
    Convert sRGB (0-255) to linear light space (0-1).
    
    WHY: sRGB has gamma encoding for display. Linear space is needed because:
    - Bilinear interpolation should happen in linear space (light adds linearly)
    - Otherwise colors will be wrong after downsampling
    
    The formula reverses sRGB gamma encoding.
    """
    x = x / 255.0  # Normalize to [0, 1]
    # Inverse gamma correction: y = x/12.92 if x ≤ 0.04045, else ((x+0.055)/1.055)^2.4
    y = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return y.astype(np.float32)

def lin2srgb(y: ImageF32) -> ImageF32:
    """
    Convert linear light (0-1) back to sRGB (0-255) for display.
    
    WHY: After processing in linear space, convert back to sRGB gamma encoding
    so the image displays correctly on screens.
    """
    # Clamp values to valid range and handle potential invalid values
    y = np.clip(y, 0.0, None)
    # Gamma encoding: x = 12.92*y if y ≤ 0.0031308, else 1.055*y^(1/2.4) - 0.055
    x = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (x * 255.0).clip(0, 255).astype(np.float32)

def bilinear_kernel(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Linear (triangle) kernel for bilinear interpolation.
    
    INTUITION: For bilinear, the weight of a pixel decreases linearly with distance.
    If a pixel is 0.0 units away, weight = 1.0
    If a pixel is 0.5 units away, weight = 0.5
    If a pixel is 1.0 unit away, weight = 0.0
    Beyond 1.0 units, weight = 0.0 (doesn't contribute)
    
    This creates a triangular (linear) falloff, not a smooth curve.
    """
    ax = np.abs(x)  # Distance from sampling point
    return np.where(ax <= 1, 1 - ax, 0.0)  # Weight = 1 - distance, clamped to [0,1]

def weight_vector_bilinear(scale: int = 4) -> VecF32:
    """
    Compute bilinear weights for each pixel in a 4x4 block when downsampling 4:1.
    
    CONCRETE EXAMPLE (scale=4):
    ----------------------------
    We have a 4x4 block in the decoy image. When downsampling to 1 pixel:
    
    - OpenCV samples at position (1.5, 1.5) relative to top-left of block
      (This is the center of the block, between pixel centers)
    
    - Each pixel in the 4x4 block has a center at integer coordinates:
      Pixel[0,0] center = (0.5, 0.5)
      Pixel[1,1] center = (1.5, 1.5)  <- closest to sampling point!
      Pixel[1,2] center = (2.5, 1.5)
      etc.
    
    - The weight for each pixel depends on distance from (1.5, 1.5):
      distance[1,1] = |1.5-1.5| + |1.5-1.5| = 0.0 → weight = 1.0 (high!)
      distance[1,2] = |1.5-1.5| + |2.5-1.5| = 1.0 → weight = 0.0 (too far)
      
    Example resulting weights (approximate):
      [0.0,  0.0,  0.0,  0.0 ]
      [0.0,  0.25, 0.25, 0.0 ]  <- only center 2x2 contributes
      [0.0,  0.25, 0.25, 0.0 ]
      [0.0,  0.0,  0.0,  0.0 ]
    
    Returns: 16-element vector of weights (flattened 4x4)
    """
    # For bilinear with scale=4, we sample at position (1.5, 1.5) in the 4x4 block
    # This gives us a 2x2 kernel centered around that point
    weights = np.zeros((scale, scale), dtype=np.float32)
    
    # The sample point in the source image for pixel (i,j) in destination
    # is at (i*scale + (scale-1)/2, j*scale + (scale-1)/2)
    # For scale=4, this is (1.5, 1.5) relative to the top-left of the 4x4 block
    center = (scale - 1) / 2.0  # = 1.5 for scale=4
    
    for y in range(scale):
        for x in range(scale):
            # Distance from sampling point (1.5, 1.5) to pixel index [y, x]
            # Pixel [y, x] is centered at (x + 0.5, y + 0.5) in continuous coordinates
            # But we compute distance from integer grid position [y, x] to center 1.5
            # Example: pixel [1, 1] is at position 1, distance from center 1.5 is |1 - 1.5| = 0.5
            dy = abs(y - center)  # Distance in y from pixel index to center
            dx = abs(x - center)  # Distance in x from pixel index to center
            
            # Bilinear weight = product of 1D linear weights
            # Weight decreases linearly with distance: w = (1 - distance) if distance < 1.0, else 0
            # If both distances < 1.0, pixel contributes
            # Example: pixel [1,1] has dy=0.5, dx=0.5, weight = (1-0.5)*(1-0.5) = 0.25
            if dy < 1.0 and dx < 1.0:
                weights[y, x] = (1.0 - dy) * (1.0 - dx)
    
    # Normalize weights to sum to 1 (ensures brightness is preserved)
    weights = weights / weights.sum()
    return weights.astype(np.float32).reshape(-1)  # Flatten to 16-element vector

# ---------- luma helpers ----------

def luma_linear(img: ImageF32) -> npt.NDArray[np.float32]:
    """
    Compute perceived brightness (luma) from RGB using Rec.709 weights.
    
    WHY: Human eyes are more sensitive to green than red or blue, so we weight:
    - Red: 21.26%   (least sensitivity)
    - Green: 71.52% (most sensitivity - our eyes see green best!)
    - Blue: 7.22%   (moderate sensitivity)
    
    Returns: (H, W) array where each pixel has its brightness value
    """
    return (
        0.2126 * img[..., 0]   # Red contribution
        + 0.7152 * img[..., 1] # Green contribution (biggest!)
        + 0.0722 * img[..., 2] # Blue contribution
    ).astype(np.float32)

def bottom_luma_mask(img: ImageF32, frac: float = 0.3) -> npt.NDArray[np.bool_]:
    """
    Create a mask identifying DARK pixels where we can hide changes.
    
    INTUITION: Changes to dark pixels are harder to see than changes to bright pixels.
    If we modify a dark pixel from 10 → 15, you won't notice.
    If we modify a bright pixel from 250 → 255, you might notice.
    
    EXAMPLE:
    - Image brightness range: 0.0 (darkest) to 1.0 (brightest)
    - frac = 0.3 means we allow edits in bottom 30% of brightness range
    - If min=0.0, max=1.0, threshold = 0.0 + 0.3*(1.0-0.0) = 0.3
    - Pixels with brightness ≤ 0.3 are editable
    
    Returns: Boolean mask (True = can edit, False = don't edit)
    """
    Y = luma_linear(img)  # Compute brightness for each pixel
    y_min = float(Y.min())  # Darkest pixel in image
    y_max = float(Y.max())  # Brightest pixel in image
    # Threshold: bottom `frac` of the brightness range
    thresh = y_min + frac * (y_max - y_min)
    return (Y <= thresh)  # True for dark pixels, False for bright ones