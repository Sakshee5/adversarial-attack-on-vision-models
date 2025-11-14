from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from utils import lin2srgb

def find_largest_embeddable_rectangle(
    editable_mask: npt.NDArray[np.bool_],
    scale: int = 4,
    min_coverage: float = 0.7
) -> Tuple[int, int, int, int]:
    """Find the largest rectangle where we have enough editable pixels."""
    H, W = editable_mask.shape
    H_blocks = H // scale
    W_blocks = W // scale
    block_mask = np.zeros((H_blocks, W_blocks), dtype=np.bool_)
    
    for j in range(H_blocks):
        for i in range(W_blocks):
            y0, x0 = j * scale, i * scale
            block = editable_mask[y0:y0+scale, x0:x0+scale]
            coverage = block.sum() / (scale * scale)
            block_mask[j, i] = (coverage >= min_coverage)
    
    best_area = 0
    best_rect = (0, 0, 0, 0)
    heights = np.zeros(W_blocks, dtype=int)
    
    for j in range(H_blocks):
        for i in range(W_blocks):
            if block_mask[j, i]:
                heights[i] += 1
            else:
                heights[i] = 0
        
        area, rect = largest_rectangle_in_histogram(heights, j)
        if area > best_area:
            best_area = area
            best_rect = rect
    
    y0_blk, x0_blk, h_blk, w_blk = best_rect
    return (y0_blk * scale, x0_blk * scale, h_blk * scale, w_blk * scale)


def largest_rectangle_in_histogram(
    heights: npt.NDArray[np.int_],
    row_idx: int
) -> Tuple[int, Tuple[int, int, int, int]]:
    """Find largest rectangle in histogram."""
    stack = []
    max_area = 0
    best_rect = (0, 0, 0, 0)
    
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            area = height * (i - idx)
            if area > max_area:
                max_area = area
                width = i - idx
                best_rect = (row_idx - height + 1, idx, height, width)
            start = idx
        stack.append((start, h))
    
    for idx, height in stack:
        area = height * (len(heights) - idx)
        if area > max_area:
            max_area = area
            width = len(heights) - idx
            best_rect = (row_idx - height + 1, idx, height, width)
    
    return max_area, best_rect


def compute_optimal_background_color(
    decoy_lin: npt.NDArray[np.float32],
    editable_mask: npt.NDArray[np.bool_]
) -> str:
    """
    Compute optimal background color based on editable regions of decoy.
    
    This analyzes the color distribution in the editable (dark) regions of the decoy
    and returns a representative color that minimizes the embedding effort.
    
    Args:
        decoy_lin: Decoy image in linear RGB space (H, W, 3)
        editable_mask: Boolean mask of editable pixels (H, W)
    
    Returns:
        Hex color string (e.g., '#1a2b3c')
    """
    # Extract editable pixels (where mask is True)
    editable_pixels = decoy_lin[editable_mask]
    
    if len(editable_pixels) == 0:
        # Fallback to default dark grey
        return '#333333'
    
    # Compute median color in editable regions (more robust than mean against outliers)
    median_color_lin = np.median(editable_pixels, axis=0)
    
    # Convert to sRGB for hex representation
    median_srgb = lin2srgb(median_color_lin.reshape(1, 1, 3)).flatten().astype(np.uint8)
    
    # Convert to hex
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        median_srgb[0], median_srgb[1], median_srgb[2]
    )
    
    return hex_color


def create_coverage_heatmap(
    editable_mask: npt.NDArray[np.bool_],
    scale: int = 4
) -> np.ndarray:
    """Create a heatmap showing editable pixel coverage per block."""
    H, W = editable_mask.shape
    H_blocks = H // scale
    W_blocks = W // scale
    heatmap = np.zeros((H_blocks, W_blocks), dtype=np.float32)
    
    for j in range(H_blocks):
        for i in range(W_blocks):
            y0, x0 = j * scale, i * scale
            block = editable_mask[y0:y0+scale, x0:x0+scale]
            heatmap[j, i] = block.sum() / (scale * scale)
    
    # Resize to match original image size for visualization
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Convert to RGB heatmap (red = low coverage, green = high coverage)
    heatmap_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    heatmap_rgb[..., 0] = ((1 - heatmap_resized) * 255).astype(np.uint8)  # Red for low
    heatmap_rgb[..., 1] = (heatmap_resized * 255).astype(np.uint8)  # Green for high
    
    return heatmap_rgb


def load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Load font with fallbacks."""
    try:
        return ImageFont.truetype("Arial.ttf", font_size)
    except OSError:
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:\\Windows\\Fonts\\arial.ttf"
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except OSError:
                    continue
        return ImageFont.load_default()


def wrap_text_to_fit(text: str, font: ImageFont.FreeTypeFont, draw: ImageDraw.Draw, max_width: int) -> List[str]:
    """Wrap text to fit within specified width."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def calculate_optimal_font_size(
    text: str,
    placement_rect: Tuple[int, int, int, int],
    scale: int = 4,
    max_font_size: int = 64,
    min_font_size: int = 8
) -> Tuple[int, bool, List[str]]:
    """Calculate optimal font size to fit text in placement area.
    
    Args:
        text: Text to render
        placement_rect: Rectangle for text placement (y0, x0, height, width) in full resolution
        scale: Downsampling scale factor
        max_font_size: Maximum font size to try
        min_font_size: Minimum acceptable font size
    
    Returns:
        Tuple of (font_size, text_fits, wrapped_lines)
    """
    y0, x0, rect_height, rect_width = placement_rect
    
    # Target dimensions (downsampled)
    target_height = rect_height // scale
    target_width = rect_width // scale
    
    # Create temporary image for text measurement
    temp_image = Image.new('RGB', (target_width, target_height), color='#000000')
    draw = ImageDraw.Draw(temp_image)
    
    # Define text area with margin
    margin = max(10, min(target_width, target_height) // 20)
    text_area_width = target_width - 2 * margin
    text_area_height = target_height - 2 * margin
    
    # Auto-calculate optimal font size - start large and reduce until it fits
    font_size = min(max_font_size, target_width, target_height)  # Cap at max_font_size for readability
    text_fits = False
    wrapped_lines = []
    
    while font_size >= min_font_size:
        font = load_font(font_size)
        wrapped_lines = wrap_text_to_fit(text, font, draw, text_area_width)
        
        if not wrapped_lines:
            font_size -= 2
            continue
        
        sample_bbox = draw.textbbox((0, 0), "Ay", font=font)
        line_height = sample_bbox[3] - sample_bbox[1]
        total_height = len(wrapped_lines) * line_height
        
        if total_height <= text_area_height:
            text_fits = True
            break
        
        font_size -= 2
    
    return font_size, text_fits, wrapped_lines


def create_text_image(
    text: str,
    full_size: Tuple[int, int],
    placement_rect: Tuple[int, int, int, int],
    font_size: int,
    scale: int = 4,
    base_image: np.ndarray = None,
    background_color: str = '#333333',
    text_color: str = '#00b002',
    tight_bbox_only: bool = False
) -> Tuple[np.ndarray, dict]:
    """Create text image with specified font size.
    
    Args:
        text: Text to render
        full_size: Full resolution size (H, W) before downsampling
        placement_rect: Rectangle for text placement (y0, x0, height, width) in full resolution
        font_size: Font size to use for text rendering
        scale: Downsampling scale factor
        base_image: Optional base image to overlay text on (downsampled resolution). 
                   If None, uses solid background_color.
        background_color: Background color for text region (hex string like '#1a2b3c')
        text_color: Color for text (ignored if using red-channel-only mode)
        tight_bbox_only: If True, only draw background around tight text bbox, not entire placement_rect
    """
    H_full, W_full = full_size
    y0, x0, rect_height, rect_width = placement_rect
    
    # Target dimensions (downsampled)
    target_height = rect_height // scale
    target_width = rect_width // scale
    
    # Create full-size target image
    if base_image is not None:
        # Use provided base image (downsampled original)
        image = Image.fromarray(base_image)
    else:
        # Fallback to solid color background
        image = Image.new('RGB', (W_full // scale, H_full // scale), color=background_color)
    
    draw = ImageDraw.Draw(image)
    
    # Define text area with margin
    margin = max(10, min(target_width, target_height) // 20)
    text_area_width = target_width - 2 * margin
    text_area_height = target_height - 2 * margin
    
    # Load font and wrap text
    font = load_font(font_size)
    wrapped_lines = wrap_text_to_fit(text, font, draw, text_area_width)
    text_fits = len(wrapped_lines) > 0
    
    # Draw text in the placement rectangle
    target_y0 = y0 // scale
    target_x0 = x0 // scale
    
    if text_fits and wrapped_lines:
        # Parse background color and create red-channel-only version
        # We want subtle contrast: small red channel difference that's invisible at high-res but visible when downsampled
        bg_r, bg_g, bg_b = int(background_color[1:3], 16), int(background_color[3:5], 16), int(background_color[5:7], 16)
        
        # Background: Red=0 (dark), keep G/B from optimal color
        text_bg_color = (0, bg_g, bg_b)
        
        # Text: Use SUBTLE red value (not 255!) - just enough to be visible when downsampled
        # The key: small difference at high-res → invisible, accumulates to visible when downsampled
        # Typical: 60-100 is subtle but effective. Higher = more visible at high-res, lower = less visible when downsampled
        subtle_red_value = 80  # Experiment with 60-120 range
        subtle_red_text = (subtle_red_value, 0, 0)  # Subtle red, not pure red!
        
        sample_bbox = draw.textbbox((0, 0), "Ay", font=font)
        line_height = sample_bbox[3] - sample_bbox[1]
        text_ascent = sample_bbox[1]
        total_height = len(wrapped_lines) * line_height
        
        # Center text vertically in the rectangle
        start_y = target_y0 + margin + (text_area_height - total_height) // 2 - text_ascent
        
        # Compute tight bounding box if requested
        if tight_bbox_only:
            # Calculate actual text bounds
            min_x, min_y = float('inf'), start_y
            max_x, max_y = 0, start_y + total_height
            
            for i, line in enumerate(wrapped_lines):
                y = start_y + i * line_height
                bbox = draw.textbbox((0, y), line, font=font)
                line_width = bbox[2] - bbox[0]
                x = target_x0 + margin + (text_area_width - line_width) // 2
                min_x = min(min_x, x)
                max_x = max(max_x, x + line_width)
            
            # Add small padding around text
            padding = 5
            tight_x0 = max(target_x0, int(min_x) - padding)
            tight_y0 = max(target_y0, int(min_y) - padding)
            tight_x1 = min(target_x0 + target_width, int(max_x) + padding)
            tight_y1 = min(target_y0 + target_height, int(max_y) + padding)
            
            # Draw background only around tight text region
            draw.rectangle(
                [tight_x0, tight_y0, tight_x1, tight_y1],
                fill=text_bg_color
            )
        else:
            # Draw background for entire placement rectangle
            draw.rectangle(
                [target_x0, target_y0, target_x0 + target_width, target_y0 + target_height],
                fill=text_bg_color
            )
        
        # Draw text
        for i, line in enumerate(wrapped_lines):
            y = start_y + i * line_height
            if y + line_height > target_y0 + target_height - margin:
                break
            
            bbox = draw.textbbox((0, y), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = target_x0 + margin + (text_area_width - line_width) // 2
            
            # Use subtle red for text (small red channel value → invisible at high-res, visible when downsampled)
            draw.text((x, y), line, font=font, fill=subtle_red_text)
    
    info = {
        'target_rect': (target_y0, target_x0, target_height, target_width),
        'highres_rect': placement_rect,
        'font_size': font_size,
        'text_fits': text_fits,
        'num_lines': len(wrapped_lines) if wrapped_lines else 0,
        'text_area': f"{target_width}×{target_height}px"
    }
    
    return np.array(image), info