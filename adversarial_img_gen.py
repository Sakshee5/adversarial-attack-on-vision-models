"""
BILINEAR ADVERSARIAL EMBEDDING - EXPLAINED
==========================================

THE BIG PICTURE:
----------------
We want to hide text/patterns in a high-resolution image such that:
1. When you look at the high-res image, you see a normal image (decoy)
2. When you downscale it 4:1 using bilinear interpolation, the text appears

EXAMPLE WITH CONCRETE NUMBERS:
-------------------------------
Let's say we have an 8x8 pixel image (decoy) and want a 2x2 target (target).

For 4:1 downsampling:
- Each target pixel (i,j) corresponds to a 4x4 block in the decoy
- Target pixel (0,0) = decoy pixels [0:4, 0:4] (top-left 4x4 block)
- Target pixel (0,1) = decoy pixels [0:4, 4:8] (top-right 4x4 block)
- etc.

If we want target[0,0] to be RED (255 in red channel), we need to modify
the decoy's 4x4 block so that when bilinear downsampling happens, we get 255.

HOW BILINEAR DOWNSAMPLING WORKS:
---------------------------------
When downsampling a 4x4 block to 1 pixel using bilinear:
- OpenCV samples at position (1.5, 1.5) relative to the top-left of the block
- This is between pixel centers, so it interpolates from nearby pixels
- For a 4x4 block, only the center 2x2 pixels contribute significantly
- The contribution (weight) of each pixel depends on its distance from (1.5, 1.5)

Example weights for 4x4 block (normalized):
    [0.0, 0.0,  0.0,  0.0 ]   # Row 0: too far, weight ≈ 0
    [0.0, 0.25, 0.25, 0.0 ]   # Row 1: center pixels get high weight
    [0.0, 0.25, 0.25, 0.0 ]   # Row 2: center pixels get high weight
    [0.0, 0.0,  0.0,  0.0 ]   # Row 3: too far, weight ≈ 0

The downsampled value = sum(pixel_value[i,j] * weight[i,j]) for all i,j in block

THE OPTIMIZATION PROBLEM:
--------------------------
For each target pixel (i,j):
1. Compute what the downsampled value WOULD be: current_value = sum(decoy_pixels * weights)
2. Compute the difference: diff = target[i,j] - current_value
3. Modify the decoy pixels to make current_value = target[i,j]

But we want to:
- Minimize the changes (least-squares: minimize sum of pixel changes squared)
- Preserve the mean of the block (don't make it look different)
- Only modify dark pixels (harder to see changes)

So we solve:
  minimize: ||delta||² + λ² * (sum of delta)²
  subject to: sum(weights * delta) = diff
              delta can only be non-zero on dark pixels

This gives us the optimal pixel modifications!
"""

from __future__ import annotations

import argparse
from math import log10
from typing import Tuple

import numpy as np
import numpy.typing as npt
import cv2
from utils import srgb2lin, lin2srgb, weight_vector_bilinear, bottom_luma_mask

ImageF32 = npt.NDArray[np.float32]
VecF32 = npt.NDArray[np.float32]

def embed_bilinear(
    decoy: ImageF32,
    target: ImageF32,
    lam: float = 0.25,
    eps: float = 0.0,
    gamma_target: float = 1.0,
    dark_frac: float = 0.3,
) -> ImageF32:
    """
    MAIN EMBEDDING FUNCTION - The Core Algorithm
    =============================================
    
    GOAL: Modify `decoy` (high-res) so that when bilinear downscaled 4:1, it matches `target`.
    
    CONCRETE EXAMPLE WALKTHROUGH:
    ---------------------------
    Let's say we have:
    - Decoy: 8x8 image (64 pixels)
    - Target: 2x2 image (4 pixels)
    - We want target[0,0] to be RED=255, others normal
    
    STEP 1: For target pixel (0,0), look at decoy's corresponding 4x4 block
       Decoy block = pixels [0:4, 0:4]  (top-left 4x4 block)
    
    STEP 2: Compute what the downsampled value WOULD be currently:
       current_value = sum(decoy_pixels * weights) for the 4x4 block
       Example: current_value = 50 (a dark red)
    
    STEP 3: Compare with target:
       target_value = 255 (bright red)
       diff = 255 - 50 = 205 (we need to increase by 205!)
    
    STEP 4: Figure out which pixels we can modify:
       - Check which pixels in the 4x4 block are DARK (using luma mask)
       - Maybe only pixels [1,1] and [2,2] are dark enough to edit
    
    STEP 5: Solve optimization problem:
       We want to modify the editable pixels to make:
       sum(modified_pixels * weights) = target_value
       
       But we also want:
       - Minimize changes (least-squares: ||delta||²)
       - Preserve block mean (don't make it look weird)
       - Only modify dark pixels (harder to see)
    
    STEP 6: Apply the computed changes to the decoy block
    STEP 7: Repeat for all target pixels
    
    PARAMETERS:
    -----------
    lam (λ): Mean-preservation weight
        - λ=0: Only modify pixels that directly affect downsampling
        - λ→∞: Preserve the mean of the block (spread changes uniformly)
        - λ=0.25: Balance between precision and preservation
    
    eps (ε): Null-space dither magnitude
        - Adds random perturbations that DON'T change the downsampled result
        - Makes the pattern harder to detect visually
        - ε=0 means no dithering
    
    dark_frac: Fraction of brightness range considered "dark" (editable)
        - 0.3 means bottom 30% of brightness range can be modified
        - Higher = can edit more pixels but might be more visible
    
    gamma_target: Pre-emphasis on target before matching
        - Usually 1.0 (no change)
        - Can help match perceptual brightness
    """
    s = 4  # Scale factor: 4x4 block → 1 pixel
    w_full: VecF32 = weight_vector_bilinear(s)  # 16 weights for the 4x4 block
    sum_w_full = float(w_full.sum())  # Should be 1.0 (normalized)

    # Precompute which pixels in the decoy can be edited (based on darkness)
    # This is a boolean mask: True = can edit, False = cannot edit
    editable_mask = bottom_luma_mask(decoy, frac=dark_frac)

    adv = decoy.copy()  # Start with a copy of the decoy
    tgt = (target ** gamma_target).astype(np.float32)  # Apply gamma to target

    H_t, W_t, _ = tgt.shape  # Target dimensions (e.g., 2x2)
    
    # MAIN LOOP: Process each target pixel one at a time
    for j in range(H_t):  # For each row in target
        for i in range(W_t):  # For each column in target
            
            # Find the corresponding 4x4 block in the decoy
            # Target pixel (i,j) corresponds to decoy pixels [j*s:(j+1)*s, i*s:(i+1)*s]
            # Example: target[0,0] → decoy[0:4, 0:4]
            y0, x0 = j * s, i * s
            blk = adv[y0:y0 + s, x0:x0 + s]  # Extract the 4x4 block
            blk_mask = editable_mask[y0:y0 + s, x0:x0 + s]  # Which pixels can we edit?

            # Flatten the mask to find indices of editable pixels
            mask_flat = blk_mask.reshape(-1)  # [True, False, True, ...] (16 values)
            idx = np.flatnonzero(mask_flat)  # Indices where mask is True: [0, 2, 5, ...]
            
            if idx.size == 0:
                continue  # No editable pixels in this block, skip it

            # Process only red channel (channel 0) to create red-on-black text effect
            # Note: Could modify all channels for multi-color embedding
            for c in (0,):  # Only modify red channel (channel 0)
                
                # STEP 1: Compute current downsampled value
                # This is what OpenCV would compute: sum(pixel_values * weights)
                # Example: if block has values [50, 60, 70, 80, ...] and weights [0.25, 0.25, 0.25, 0.25, ...]
                #          current = 50*0.25 + 60*0.25 + 70*0.25 + 80*0.25 = 65.0
                blk_flat = blk[..., c].reshape(-1)  # Flatten 4x4 block to 16 values
                y_cur = float((w_full * blk_flat).sum())  # Weighted sum = current downsampled value
                
                # STEP 2: Compute how much we need to change
                # diff = target_value - current_value
                # Example: if target=255 and current=50, diff=205 (need big increase)
                diff = float(tgt[j, i, c] - y_cur)

                # STEP 3: Restrict to editable subset of pixels
                # We can only modify pixels where blk_mask is True
                w_sub = w_full[idx]  # Weights for editable pixels only
                M = float(w_sub.size)  # Number of editable pixels (e.g., 4 pixels)

                # STEP 4: Recompute solver terms for the editable subset
                # These are needed for the optimization formula
                sum_w_sub = float(w_sub.sum())  # Sum of weights for editable pixels
                w_norm2_sub = float(w_sub @ w_sub)  # Sum of squares of weights

                # STEP 5: Solve the optimization problem
                # We're solving: minimize ||delta||² + λ²(sum delta)² subject to sum(weights*delta) = diff
                # 
                # The closed-form solution (derived via Lagrange multipliers) is:
                #   delta[i] = diff * (M * w[i] - λ * sum_w) / denom
                #   where denom = M*w_norm² + λ² - sum_w²
                #
                # INTUITION:
                # - If λ=0: Changes are proportional to weights (pixels with high weight change more)
                # - If λ→∞: Changes spread uniformly (preserve mean)
                # - The formula balances these two goals
                denom = (M * w_norm2_sub + lam**2) - (sum_w_sub ** 2)
                if abs(denom) < 1e-12:
                    continue  # Division by zero (ill-conditioned), skip this block

                # Compute the optimal pixel modifications
                delta_sub = diff * (M * w_sub - lam * sum_w_sub) / denom
                # delta_sub is now a vector of changes, one per editable pixel
                # Example: delta_sub = [51.25, 51.25, 51.25, 51.25] (if 4 pixels editable)

                # STEP 6: Optional null-space dithering
                # This adds random noise that DOESN'T change the downsampled result
                # Makes the pattern harder to detect visually
                if eps > 0.0 and w_sub.size >= 3:
                    # Null space = changes that keep both weighted sum AND mean unchanged
                    # Found using SVD (Singular Value Decomposition)
                    C_sub = np.vstack([w_sub, np.ones_like(w_sub, dtype=np.float32)])  # Constraint matrix
                    _, _, Vh_sub = np.linalg.svd(C_sub, full_matrices=True)  # Decompose
                    B_sub = Vh_sub[2:].astype(np.float32)  # Null space basis (orthogonal to constraints)
                    if B_sub.size > 0:
                        # Add random dither in null space (doesn't affect result!)
                        random_dither = np.random.randn(B_sub.shape[0]).astype(np.float32)
                        null_space_delta = (B_sub.T @ random_dither).astype(np.float32)
                        delta_sub = delta_sub + eps * null_space_delta

                # STEP 7: Apply changes to the 4x4 block
                # Create a full 16-element vector, fill in editable positions
                delta_vec = np.zeros_like(w_full, dtype=np.float32)  # [0, 0, 0, ...] (16 zeros)
                delta_vec[idx] = delta_sub.astype(np.float32)  # Fill editable positions: [51.25, 0, 51.25, ...]
                blk[..., c] = blk[..., c] + delta_vec.reshape(s, s)  # Add changes to block

            # Save the modified block back to the adversarial image
            adv[y0:y0 + s, x0:x0 + s] = blk

    return adv.astype(np.float32)  # Return the modified decoy

def mse_psnr(a: ImageF32, b: ImageF32) -> Tuple[float, float]:
    """
    Compute Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).
    
    MSE: Average squared difference between pixels
      - Lower is better (0 = perfect match)
      - Example: MSE = 0.01 means average error of 0.1 (in normalized space)
    
    PSNR: Quality metric in decibels (dB)
      - Higher is better
      - Typical: 30-40 dB = good quality, 20-30 dB = acceptable, <20 dB = poor
      - Formula: PSNR = 10 * log10(1 / MSE)
      - If MSE = 0.01, PSNR = 10 * log10(100) = 20 dB
    """
    mse = float(np.mean((a - b) ** 2))  # Average squared error
    psnr = float("inf") if mse == 0 else 10.0 * log10(1.0 / mse)  # PSNR in dB
    return mse, psnr