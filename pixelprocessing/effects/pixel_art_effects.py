import numpy as np
import logging
from scipy import ndimage
import cv2

def enhance_outlines(image_array, outline_color=None, thickness=1):
    """
    Enhance pixel art by detecting and improving outlines.
    
    Args:
        image_array: Input image as numpy array
        outline_color: RGB color for outlines (if None, uses darkest color in image)
        thickness: Outline thickness in pixels (1 or 2)
        
    Returns:
        Enhanced image with improved outlines as numpy array
    """
    logger = logging.getLogger(__name__)
    
    try:
        height, width = image_array.shape[:2]
        enhanced = np.copy(image_array)
        
        # If no outline color specified, use the darkest color in the image
        if outline_color is None:
            # Calculate luminance of each unique color
            unique_colors = np.unique(image_array.reshape(-1, 3), axis=0)
            luminance = np.sum(unique_colors * np.array([0.299, 0.587, 0.114]), axis=1)
            darkest_idx = np.argmin(luminance)
            outline_color = unique_colors[darkest_idx]
            logger.info(f"Using darkest color as outline: RGB{tuple(outline_color)}")
        
        # Convert outline color to numpy array if it isn't already
        outline_color = np.array(outline_color)
        
        # Create a binary mask for edges
        # Convert to grayscale for edge detection
        gray = np.dot(image_array, [0.299, 0.587, 0.114]).astype(np.uint8)
        
        # Apply edge detection filter
        sobel_h = ndimage.sobel(gray, axis=0)
        sobel_v = ndimage.sobel(gray, axis=1)
        edges = np.sqrt(sobel_h**2 + sobel_v**2) > 30
        
        # Apply detected edges to the image using outline color
        for y in range(height):
            for x in range(width):
                if edges[y, x]:
                    enhanced[y, x] = outline_color
        
        # If thickness is 2, dilate the edges
        if thickness == 2:
            # Create a copy of the edges
            edges_dilated = ndimage.binary_dilation(edges)
            
            # Apply dilated edges without overwriting original edges
            for y in range(height):
                for x in range(width):
                    if edges_dilated[y, x] and not edges[y, x]:
                        enhanced[y, x] = outline_color
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing pixel art outlines: {str(e)}")
        # Return the original image if enhancement fails
        return image_array

def upscale_pixel_art(image_array, scale=2, method='nearest'):
    """
    Scale up pixel art using various methods.
    
    Args:
        image_array: Input image as numpy array
        scale: Integer scale factor (2, 3, 4, etc.)
        method: Scaling method ('nearest', 'hq2x', 'eagle')
        
    Returns:
        Upscaled image as numpy array
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure scale is an integer
        scale = int(scale)
        if scale < 1:
            logger.warning(f"Invalid scale factor {scale}, using 2x")
            scale = 2
        
        height, width = image_array.shape[:2]
        
        if method == 'nearest':
            # Nearest neighbor (pixel-perfect) scaling
            from PIL import Image
            pil_image = Image.fromarray(image_array)
            upscaled = pil_image.resize((width * scale, height * scale), Image.NEAREST)
            return np.array(upscaled)
            
        elif method == 'hq2x':
            # HQ2X algorithm (simplified version)
            # Only works correctly for 2x scale
            if scale != 2:
                logger.warning(f"HQ2X only supports 2x scaling, using nearest for {scale}x")
                return upscale_pixel_art(image_array, scale, 'nearest')
            
            # Create output array
            upscaled = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
            
            # Process each pixel
            for y in range(height):
                for x in range(width):
                    # Get the current pixel
                    p = image_array[y, x]
                    
                    # Default: Fill 2x2 block with the original pixel
                    upscaled[y*2:y*2+2, x*2:x*2+2] = p
                    
                    # Get neighboring pixels (with boundary checks)
                    nw = image_array[max(0, y-1), max(0, x-1)]
                    n = image_array[max(0, y-1), x]
                    ne = image_array[max(0, y-1), min(width-1, x+1)]
                    w = image_array[y, max(0, x-1)]
                    e = image_array[y, min(width-1, x+1)]
                    sw = image_array[min(height-1, y+1), max(0, x-1)]
                    s = image_array[min(height-1, y+1), x]
                    se = image_array[min(height-1, y+1), min(width-1, x+1)]
                    
                    # HQ2X interpolation logic
                    # Top-left pixel
                    if np.array_equal(w, n) and not np.array_equal(w, p):
                        upscaled[y*2, x*2] = w
                    
                    # Top-right pixel
                    if np.array_equal(n, e) and not np.array_equal(n, p):
                        upscaled[y*2, x*2+1] = e
                    
                    # Bottom-left pixel
                    if np.array_equal(w, s) and not np.array_equal(w, p):
                        upscaled[y*2+1, x*2] = w
                    
                    # Bottom-right pixel
                    if np.array_equal(s, e) and not np.array_equal(s, p):
                        upscaled[y*2+1, x*2+1] = e
            
            return upscaled
            
        elif method == 'eagle':
            # Eagle algorithm (simplified version)
            # Only works correctly for 2x scale
            if scale != 2:
                logger.warning(f"Eagle only supports 2x scaling, using nearest for {scale}x")
                return upscale_pixel_art(image_array, scale, 'nearest')
            
            # Create output array
            upscaled = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
            
            # Process each pixel
            for y in range(height):
                for x in range(width):
                    # Get the current pixel
                    color = image_array[y, x]
                    
                    # Fill 2x2 block with the original pixel
                    upscaled[y*2][x*2] = color
                    upscaled[y*2][x*2+1] = color
                    upscaled[y*2+1][x*2] = color
                    upscaled[y*2+1][x*2+1] = color
                    
                    # Get neighboring pixels (with boundary checks)
                    if y > 0 and x > 0:
                        if np.array_equal(image_array[y-1][x-1], image_array[y][x-1]) and \
                           np.array_equal(image_array[y-1][x-1], image_array[y-1][x]):
                            upscaled[y*2][x*2] = image_array[y-1][x-1]
                    
                    if y > 0 and x < width-1:
                        if np.array_equal(image_array[y-1][x+1], image_array[y-1][x]) and \
                           np.array_equal(image_array[y-1][x+1], image_array[y][x+1]):
                            upscaled[y*2][x*2+1] = image_array[y-1][x+1]
                    
                    if y < height-1 and x > 0:
                        if np.array_equal(image_array[y+1][x-1], image_array[y][x-1]) and \
                           np.array_equal(image_array[y+1][x-1], image_array[y+1][x]):
                            upscaled[y*2+1][x*2] = image_array[y+1][x-1]
                    
                    if y < height-1 and x < width-1:
                        if np.array_equal(image_array[y+1][x+1], image_array[y][x+1]) and \
                           np.array_equal(image_array[y+1][x+1], image_array[y+1][x]):
                            upscaled[y*2+1][x*2+1] = image_array[y+1][x+1]
            
            return upscaled
            
        else:
            logger.warning(f"Unknown scaling method: {method}, using nearest")
            return upscale_pixel_art(image_array, scale, 'nearest')
        
    except Exception as e:
        logger.error(f"Error upscaling pixel art: {str(e)}")
        # Return the original image if upscaling fails
        return image_array

def add_pixel_grid(image_array, grid_color=(20, 20, 20), grid_alpha=0.3):
    """
    Add a pixel grid overlay to an image.
    
    Args:
        image_array: Input image as numpy array
        grid_color: RGB color for grid lines
        grid_alpha: Grid line opacity (0.0 to 1.0)
        
    Returns:
        Image with pixel grid overlay
    """
    logger = logging.getLogger(__name__)
    
    try:
        height, width = image_array.shape[:2]
        
        # Create a copy to avoid modifying the original
        result = np.copy(image_array)
        
        # Convert grid color to numpy array with alpha
        grid_color_a = np.array([*grid_color, 255]) * np.array([1, 1, 1, grid_alpha])
        
        # Create grid using vectorized operations
        for y in range(1, height):
            # Blend grid lines with original pixels
            original = result[y-1, :].astype(float)
            grid_line = np.array([*grid_color] * width).reshape(width, 3).astype(float)
            
            # Apply alpha blending
            blended = original * (1 - grid_alpha) + grid_line * grid_alpha
            result[y-1, :] = np.clip(blended, 0, 255).astype(np.uint8)
        
        for x in range(1, width):
            # Blend grid lines with original pixels
            original = result[:, x-1].astype(float)
            grid_line = np.array([*grid_color] * height).reshape(height, 3).astype(float)
            
            # Apply alpha blending
            blended = original * (1 - grid_alpha) + grid_line * grid_alpha
            result[:, x-1] = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        logger.error(f"Error adding pixel grid: {str(e)}")
        # Return the original image if effect application fails
        return image_array

def pixelate(image_array, pixel_size=8):
    """
    Pixelate an image to create a pixel art effect.
    
    Args:
        image_array: Input image as numpy array
        pixel_size: Size of pixelation blocks
        
    Returns:
        Pixelated image as numpy array
    """
    logger = logging.getLogger(__name__)
    
    try:
        height, width = image_array.shape[:2]
        
        # Calculate new dimensions
        small_h = height // pixel_size
        small_w = width // pixel_size
        
        # Resize down
        from PIL import Image
        pil_image = Image.fromarray(image_array)
        small_image = pil_image.resize((small_w, small_h), Image.BILINEAR)
        
        # Resize back up with nearest neighbor
        pixelated = small_image.resize((width, height), Image.NEAREST)
        
        return np.array(pixelated)
        
    except Exception as e:
        logger.error(f"Error pixelating image: {str(e)}")
        # Return the original image if effect application fails
        return image_array

def apply_color_banding(image_array, bands=4):
    """
    Apply color banding (posterization) effect for retro look.
    
    Args:
        image_array: Input image as numpy array
        bands: Number of color bands per channel
        
    Returns:
        Image with color banding applied
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Convert to float for processing
        float_img = image_array.astype(float) / 255.0
        
        # Apply banding
        banded = np.floor(float_img * bands) / (bands - 1)
        
        # Convert back to uint8
        result = np.clip(banded * 255.0, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        logger.error(f"Error applying color banding: {str(e)}")
        # Return the original image if effect application fails
        return image_array

def apply_pixelation_effect(image_array, pixel_size=8, sharpen=True, enhance_contrast=True):
    """
    Apply a complete pixelation effect with multiple steps.
    
    Args:
        image_array: Input image as numpy array
        pixel_size: Size of pixelation blocks
        sharpen: Whether to apply sharpening
        enhance_contrast: Whether to enhance contrast
        
    Returns:
        Processed image with pixel art effect
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Apply pixelation
        result = pixelate(image_array, pixel_size)
        
        # 2. Enhance contrast if requested
        if enhance_contrast:
            # Convert to float for processing
            float_img = result.astype(float)
            
            # Scale each channel to full range
            for c in range(3):
                channel = float_img[:, :, c]
                min_val = np.min(channel)
                max_val = np.max(channel)
                
                if max_val > min_val:
                    # Stretch to full range
                    float_img[:, :, c] = (channel - min_val) * 255.0 / (max_val - min_val)
            
            # Convert back to uint8
            result = np.clip(float_img, 0, 255).astype(np.uint8)
        
        # 3. Apply sharpening if requested
        if sharpen:
            # Define sharpening kernel
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            
            # Apply sharpening
            result = cv2.filter2D(result, -1, kernel)
        
        return result
        
    except Exception as e:
        logger.error(f"Error applying pixelation effect: {str(e)}")
        # Return the original image if effect application fails
        return image_array
