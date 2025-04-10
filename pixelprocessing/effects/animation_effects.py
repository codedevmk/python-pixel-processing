import numpy as np
from PIL import Image
import logging
import os

def create_palette_cycling(image_array, palette, cycle_range=(0, 8), 
                           frames=12, duration=100):
    """
    Create frames for a palette cycling animation.
    
    Palette cycling is a retro animation technique that shifts colors
    in the palette while keeping the image index references the same,
    creating the illusion of motion.
    
    Args:
        image_array: Indexed image as numpy array or RGB image to be converted
        palette: Color palette as numpy array
        cycle_range: Tuple of (start, end) indices in the palette to cycle
        frames: Number of frames in the animation
        duration: Duration of each frame in milliseconds
        
    Returns:
        List of numpy arrays representing animation frames
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create a mapper to convert RGB image to palette indices
        from scipy.spatial import KDTree
        
        # If the image is already indexed, convert to RGB first
        if len(image_array.shape) == 2:
            # This is an indexed image, create a RGB version
            height, width = image_array.shape
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Map each index to the corresponding palette color
            for y in range(height):
                for x in range(width):
                    idx = image_array[y, x]
                    if idx < len(palette):
                        rgb_image[y, x] = palette[idx]
            
            image_array = rgb_image
        
        # Create a mapping from RGB values to palette indices
        tree = KDTree(palette)
        
        # Find the closest palette index for each pixel
        height, width = image_array.shape[:2]
        flat_image = image_array.reshape(-1, 3)
        indices = tree.query(flat_image)[1]
        indexed_image = indices.reshape(height, width)
        
        # Validate cycle range
        start, end = cycle_range
        if start < 0 or end > len(palette) or start >= end:
            logger.warning(f"Invalid cycle range: {cycle_range}, using (0, min(8, len(palette)))")
            start = 0
            end = min(8, len(palette))
        
        logger.info(f"Creating palette cycling animation with {frames} frames, cycling colors {start}-{end}")
        
        # Create frames by shifting the palette
        frames_list = []
        cycle_length = end - start
        
        for i in range(frames):
            # Create a copy of the palette
            shifted_palette = palette.copy()
            
            # Calculate shift amount for this frame
            shift_amount = i % cycle_length
            
            # Shift the colors in the specified range
            shifted_colors = np.roll(palette[start:end], shift_amount, axis=0)
            shifted_palette[start:end] = shifted_colors
            
            # Create frame by mapping indices to shifted palette
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    idx = indexed_image[y, x]
                    if idx < len(shifted_palette):
                        frame[y, x] = shifted_palette[idx]
            
            frames_list.append(frame)
        
        return frames_list
        
    except Exception as e:
        logger.error(f"Error creating palette cycling animation: {str(e)}")
        # Return a single frame as fallback
        return [image_array]

def save_animation(frames, output_path, duration=100, loop=0):
    """
    Save animation frames as an animated GIF.
    
    Args:
        frames: List of frame arrays
        output_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
        loop: Number of times to loop (0 = infinite)
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not frames:
            logger.error("No frames provided to save_animation")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert frames to PIL Images
        pil_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                pil_frames.append(Image.fromarray(frame))
            elif isinstance(frame, Image.Image):
                pil_frames.append(frame)
            else:
                logger.warning(f"Skipping invalid frame type: {type(frame)}")
        
        if not pil_frames:
            logger.error("No valid frames to save")
            return False
        
        # Save as GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=False,  # Disable optimization to preserve palette cycling
            duration=duration,
            loop=loop
        )
        
        logger.info(f"Saved animation with {len(pil_frames)} frames to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving animation: {str(e)}")
        return False

def create_sprite_sheet_animation(sprite_sheet, frame_width, frame_height, 
                                output_path=None, duration=100, loop=0):
    """
    Create an animation from a sprite sheet.
    
    Args:
        sprite_sheet: Sprite sheet image as numpy array
        frame_width: Width of each frame in pixels
        frame_height: Height of each frame in pixels
        output_path: Path to save the GIF (optional)
        duration: Duration of each frame in milliseconds
        loop: Number of times to loop (0 = infinite)
        
    Returns:
        List of frames if output_path is None, otherwise saves GIF and returns True
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Convert to PIL Image if numpy array
        if isinstance(sprite_sheet, np.ndarray):
            sprite_sheet = Image.fromarray(sprite_sheet)
        
        # Get sprite sheet dimensions
        sheet_width, sheet_height = sprite_sheet.size
        
        # Calculate number of frames
        cols = sheet_width // frame_width
        rows = sheet_height // frame_height
        
        # Extract frames
        frames = []
        for row in range(rows):
            for col in range(cols):
                # Calculate frame position
                left = col * frame_width
                upper = row * frame_height
                right = left + frame_width
                lower = upper + frame_height
                
                # Extract frame
                frame = sprite_sheet.crop((left, upper, right, lower))
                frames.append(frame)
        
        logger.info(f"Extracted {len(frames)} frames from sprite sheet")
        
        # Save as GIF if output path is provided
        if output_path:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=duration,
                loop=loop
            )
            logger.info(f"Saved sprite sheet animation to {output_path}")
            return True
        
        # Otherwise return the frames
        return frames
        
    except Exception as e:
        logger.error(f"Error creating sprite sheet animation: {str(e)}")
        return False

def create_ping_pong_animation(frames, output_path=None, duration=100, loop=0):
    """
    Create a ping-pong animation that plays forward then backward.
    
    Args:
        frames: List of frame arrays
        output_path: Path to save the GIF (optional)
        duration: Duration of each frame in milliseconds
        loop: Number of times to loop (0 = infinite)
        
    Returns:
        List of frames if output_path is None, otherwise saves GIF and returns True
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not frames:
            logger.error("No frames provided")
            return False
        
        # Create reversed frames (excluding first and last to avoid duplication)
        if len(frames) > 2:
            ping_pong_frames = frames + frames[-2:0:-1]
        else:
            ping_pong_frames = frames + frames[::-1]
        
        # Convert to PIL Images if numpy arrays
        pil_frames = []
        for frame in ping_pong_frames:
            if isinstance(frame, np.ndarray):
                pil_frames.append(Image.fromarray(frame))
            elif isinstance(frame, Image.Image):
                pil_frames.append(frame)
            else:
                logger.warning(f"Skipping invalid frame type: {type(frame)}")
        
        # Save as GIF if output path is provided
        if output_path and pil_frames:
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=True,
                duration=duration,
                loop=loop
            )
            logger.info(f"Saved ping-pong animation to {output_path}")
            return True
        
        # Otherwise return the frames
        return ping_pong_frames if not pil_frames else pil_frames
        
    except Exception as e:
        logger.error(f"Error creating ping-pong animation: {str(e)}")
        return False

def create_fade_transition(start_frame, end_frame, num_frames=10):
    """
    Create a smooth fade transition between two images.
    
    Args:
        start_frame: Starting frame as numpy array
        end_frame: Ending frame as numpy array
        num_frames: Number of transition frames
        
    Returns:
        List of transition frames
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure shapes match
        if start_frame.shape != end_frame.shape:
            logger.warning(f"Frame shapes don't match: {start_frame.shape} vs {end_frame.shape}")
            # Resize end frame to match start frame
            from PIL import Image
            end_pil = Image.fromarray(end_frame)
            end_resized = end_pil.resize(
                (start_frame.shape[1], start_frame.shape[0]), 
                Image.BILINEAR
            )
            end_frame = np.array(end_resized)
        
        # Create transition frames
        transition_frames = []
        
        for i in range(num_frames):
            # Calculate blend factor
            alpha = i / (num_frames - 1) if num_frames > 1 else 0.5
            
            # Blend frames
            blended = (1 - alpha) * start_frame.astype(float) + alpha * end_frame.astype(float)
            transition_frames.append(blended.astype(np.uint8))
        
        return transition_frames
        
    except Exception as e:
        logger.error(f"Error creating fade transition: {str(e)}")
        return [start_frame, end_frame]
