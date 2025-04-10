import numpy as np
import logging
from scipy.ndimage import gaussian_filter

def apply_crt_effect(image_array, scanline_strength=0.2, curvature=0.1,
                     color_bleed=0.5, glow=0.2, noise=0.05):
    """
    Apply a retro CRT screen effect to an image.
    
    Args:
        image_array: Input image as numpy array (RGB)
        scanline_strength: Strength of scanline effect (0.0 to 1.0)
        curvature: Strength of screen curvature effect (0.0 to 0.5)
        color_bleed: Amount of color bleeding (0.0 to 1.0)
        glow: Strength of glow effect (0.0 to 1.0)
        noise: Amount of noise to add (0.0 to 0.5)
        
    Returns:
        Image with CRT effect applied as numpy array
    """
    logger = logging.getLogger(__name__)
    
    try:
        height, width = image_array.shape[:2]
        
        # Create output array as float for processing
        crt_effect = np.copy(image_array).astype(float)
        
        # 1. Apply scanlines
        if scanline_strength > 0:
            for y in range(height):
                if y % 2 == 0:  # Apply to every other line
                    crt_effect[y, :] *= (1.0 - scanline_strength)
        
        # 2. Apply color bleeding (slight horizontal blur with channel offset)
        if color_bleed > 0:
            # Apply different blur strengths to each color channel
            for c in range(3):
                sigma = 0.5 + (c * 0.25)  # Increasing sigma for R -> G -> B
                crt_effect[:, :, c] = gaussian_filter(
                    crt_effect[:, :, c], 
                    sigma=(0, sigma * color_bleed)
                )
            
            # Offset red channel slightly to the right
            if width > 3:
                offset = max(1, int(width * 0.01 * color_bleed))
                crt_effect[:, offset:, 0] = crt_effect[:, :-offset, 0]
        
        # 3. Apply glow effect (overall blur)
        if glow > 0:
            blur_sigma = 1.0 * glow
            glow_image = gaussian_filter(crt_effect, sigma=(blur_sigma, blur_sigma, 0))
            crt_effect = crt_effect * 0.8 + glow_image * 0.2
        
        # 4. Apply vignette/curvature effect
        if curvature > 0:
            y, x = np.ogrid[:height, :width]
            center_y, center_x = height / 2, width / 2
            y = 2 * (y - center_y) / height
            x = 2 * (x - center_x) / width
            
            # Calculate distance from center (normalized)
            r = np.sqrt(x**2 + y**2)
            
            # Apply curvature darkening
            vignette = 1.0 - np.clip(r - curvature, 0, 1) / (1 - curvature)
            crt_effect = crt_effect * vignette[:, :, np.newaxis]
        
        # 5. Add noise
        if noise > 0:
            noise_amount = noise * 50  # Scale to reasonable range
            random_noise = np.random.normal(0, noise_amount, crt_effect.shape)
            crt_effect += random_noise
        
        # Convert back to uint8
        crt_effect = np.clip(crt_effect, 0, 255).astype(np.uint8)
        
        return crt_effect
        
    except Exception as e:
        logger.error(f"Error applying CRT effect: {str(e)}")
        # Return the original image if effect application fails
        return image_array

def apply_phosphor_afterglow(frames, decay=0.7, frames_to_blend=3):
    """
    Apply phosphor afterglow effect to an animation.
    
    This simulates the slow decay of phosphors in old CRT monitors,
    creating a trailing effect for moving objects.
    
    Args:
        frames: List of frame arrays
        decay: Decay factor (0.0 to 1.0)
        frames_to_blend: Number of previous frames to blend
        
    Returns:
        List of processed frame arrays
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not frames:
            return frames
            
        # Make a copy of the frames to avoid modifying originals
        result_frames = [np.copy(frame) for frame in frames]
        n_frames = len(frames)
        
        # Process each frame (except the first)
        for i in range(1, n_frames):
            # Start with the current frame
            current = result_frames[i].astype(float)
            
            # Blend with previous frames
            for j in range(1, min(frames_to_blend + 1, i + 1)):
                # Calculate decay factor for this historical frame
                frame_decay = decay ** j
                
                # Blend with the historical frame
                previous = result_frames[i - j].astype(float)
                current = np.maximum(current, previous * frame_decay)
            
            # Update the result
            result_frames[i] = np.clip(current, 0, 255).astype(np.uint8)
        
        return result_frames
        
    except Exception as e:
        logger.error(f"Error applying phosphor afterglow: {str(e)}")
        # Return the original frames if effect application fails
        return frames

def add_screen_curvature(image_array, curvature=0.1):
    """
    Add screen curvature distortion to an image.
    
    This simulates the curved screen of old CRT monitors.
    
    Args:
        image_array: Input image as numpy array
        curvature: Strength of curvature effect (0.0 to 0.5)
        
    Returns:
        Image with curvature effect applied
    """
    logger = logging.getLogger(__name__)
    
    try:
        import cv2
        
        height, width = image_array.shape[:2]
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Calculate normalized coordinates
        x_norm = 2 * (x / width - 0.5)
        y_norm = 2 * (y / height - 0.5)
        
        # Apply distortion
        r = np.sqrt(x_norm**2 + y_norm**2)
        distortion = 1 + curvature * r**2
        
        x_distorted = x_norm * distortion
        y_distorted = y_norm * distortion
        
        # Convert back to pixel coordinates
        x_output = (x_distorted * 0.5 + 0.5) * width
        y_output = (y_distorted * 0.5 + 0.5) * height
        
        # Create map for remap function
        map_x = x_output.astype(np.float32)
        map_y = y_output.astype(np.float32)
        
        # Apply remapping
        curved = cv2.remap(image_array, map_x, map_y, cv2.INTER_LINEAR)
        
        # Create a mask for the valid region
        mask = np.ones_like(curved)
        mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)
        
        # Apply mask to remove artifacts at the edges
        valid_area = (mask > 0.9)
        curved = curved * valid_area
        
        return curved
        
    except Exception as e:
        logger.error(f"Error applying screen curvature: {str(e)}")
        # Return the original image if effect application fails
        return image_array
