import numpy as np
from PIL import Image
import logging
from scipy import ndimage
import cv2

class BackgroundRemover:
    """
    Handles background removal for images, creating transparent PNGs.
    
    This class provides various methods to detect and remove backgrounds
    from images, producing transparent PNG images with an alpha channel.
    """
    
    def __init__(self, config):
        """
        Initialize the background remover.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def remove_background(self, image_array, method='color', color_threshold=30, 
                          corner_sample=True, bg_color=None, feather_edges=False):
        """
        Remove background from an image.
        
        Args:
            image_array: Input image as numpy array (RGB)
            method: Method for background removal ('color', 'contrast', 'edge')
            color_threshold: Threshold for color distance detection
            corner_sample: Whether to sample corners to detect background color
            bg_color: Background color as RGB tuple, if provided
            feather_edges: Whether to soften edges of the mask
            
        Returns:
            RGBA image array with transparent background
        """
        if method == 'color':
            return self._remove_bg_by_color(
                image_array, color_threshold, corner_sample, bg_color, feather_edges)
        elif method == 'contrast':
            return self._remove_bg_by_contrast(image_array, color_threshold, feather_edges)
        elif method == 'edge':
            return self._remove_bg_by_edge(image_array, color_threshold, feather_edges)
        else:
            self.logger.warning(f"Unknown background removal method: {method}, falling back to color")
            return self._remove_bg_by_color(
                image_array, color_threshold, corner_sample, bg_color, feather_edges)
    
    def _remove_bg_by_color(self, image_array, threshold=30, corner_sample=True, 
                           bg_color=None, feather_edges=False):
        """
        Remove background based on color similarity.
        
        If bg_color is None and corner_sample is True, samples the corners of the image
        to determine the likely background color.
        
        Args:
            image_array: Input image as numpy array (RGB)
            threshold: Color distance threshold
            corner_sample: Whether to sample corners to detect background
            bg_color: Background color as RGB tuple, if provided
            feather_edges: Whether to soften edges of the mask
            
        Returns:
            RGBA image array with transparent background
        """
        try:
            height, width = image_array.shape[:2]
            
            # Create output array with alpha channel
            output = np.zeros((height, width, 4), dtype=np.uint8)
            output[:, :, :3] = image_array  # Copy RGB channels
            
            # Determine background color
            if bg_color is None and corner_sample:
                # Sample corners to determine likely background color
                corner_pixels = [
                    image_array[0, 0],          # Top-left
                    image_array[0, width-1],    # Top-right
                    image_array[height-1, 0],   # Bottom-left
                    image_array[height-1, width-1]  # Bottom-right
                ]
                bg_color = np.median(corner_pixels, axis=0).astype(np.uint8)
                self.logger.info(f"Detected background color: RGB{tuple(bg_color)}")
            elif bg_color is None:
                # Without corner sampling, use most frequent color in the image
                pixels = image_array.reshape(-1, 3)
                pixels_tuple = [tuple(pixel) for pixel in pixels]
                
                # Find most common color
                from collections import Counter
                color_counts = Counter(pixels_tuple)
                bg_color = np.array(color_counts.most_common(1)[0][0])
                self.logger.info(f"Using most frequent color as background: RGB{tuple(bg_color)}")
            
            # Convert bg_color to numpy array if it's not already
            bg_color = np.array(bg_color)
            
            # Calculate color distance from every pixel to background color
            flat_image = image_array.reshape(-1, 3).astype(np.int32)
            distances = np.sqrt(np.sum(np.square(flat_image - bg_color), axis=1))
            
            # Create alpha mask based on distance
            alpha_flat = np.where(distances <= threshold, 0, 255)
            alpha = alpha_flat.reshape(height, width)
            
            # Option to feather edges for smoother transitions
            if feather_edges:
                # Create a distance field from the binary mask
                distance = ndimage.distance_transform_edt(alpha)
                
                # Normalize and apply sigmoid curve for smooth falloff
                max_dist = min(10, np.max(distance))  # Limit feather radius
                if max_dist > 0:
                    feathered = 255 * (1 - np.exp(-distance / (max_dist / 3)))
                    alpha = np.clip(feathered, 0, 255).astype(np.uint8)
            
            # Apply alpha channel
            output[:, :, 3] = alpha
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error removing background by color: {str(e)}")
            # Return original image with full alpha
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, :3] = image_array
            rgba[:, :, 3] = 255
            return rgba
    
    def _remove_bg_by_contrast(self, image_array, threshold=30, feather_edges=False):
        """
        Remove background based on contrast with foreground.
        
        This method works well when the foreground has more detail/contrast than the background.
        
        Args:
            image_array: Input image as numpy array (RGB)
            threshold: Contrast threshold
            feather_edges: Whether to soften edges of the mask
            
        Returns:
            RGBA image array with transparent background
        """
        try:
            height, width = image_array.shape[:2]
            
            # Create output array with alpha channel
            output = np.zeros((height, width, 4), dtype=np.uint8)
            output[:, :, :3] = image_array  # Copy RGB channels
            
            # Convert to grayscale
            gray = np.dot(image_array, [0.299, 0.587, 0.114]).astype(np.uint8)
            
            # Calculate local contrast (gradient magnitude)
            sobel_h = ndimage.sobel(gray, axis=0)
            sobel_v = ndimage.sobel(gray, axis=1)
            magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
            
            # Normalize contrast to 0-255
            magnitude = magnitude * (255.0 / (magnitude.max() or 1.0))
            
            # Create binary mask based on contrast
            mask = np.where(magnitude > threshold, 255, 0).astype(np.uint8)
            
            # Fill holes in the mask
            mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
            
            # Option to feather edges
            if feather_edges:
                kernel_size = max(3, min(9, int(min(height, width) / 20)))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
            
            # Apply alpha channel
            output[:, :, 3] = mask
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error removing background by contrast: {str(e)}")
            # Return original image with full alpha
            rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image_array
            rgba[:, :, 3] = 255
            return rgba
    
    def _remove_bg_by_edge(self, image_array, threshold=30, feather_edges=False):
        """
        Remove background using edge detection and flood fill.
        
        This method works well for images with clear boundaries between foreground and background.
        
        Args:
            image_array: Input image as numpy array (RGB)
            threshold: Edge detection threshold
            feather_edges: Whether to soften edges of the mask
            
        Returns:
            RGBA image array with transparent background
        """
        try:
            height, width = image_array.shape[:2]
            
            # Create output array with alpha channel
            output = np.zeros((height, width, 4), dtype=np.uint8)
            output[:, :, :3] = image_array  # Copy RGB channels
            
            # Convert to grayscale
            gray = np.dot(image_array, [0.299, 0.587, 0.114]).astype(np.uint8)
            
            # Detect edges
            edges = cv2.Canny(gray, threshold, threshold * 2)
            
            # Dilate edges to close small gaps
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Create mask from edges using flood fill
            # Start with all white
            mask = np.ones((height + 2, width + 2), np.uint8) * 255
            
            # Define corners as seeds for flood fill
            seeds = [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)]
            
            for seed in seeds:
                # If the pixel at this corner is not an edge, flood fill from here
                if edges[min(seed[0], height-1), min(seed[1], width-1)] == 0:
                    cv2.floodFill(edges, mask, seed, 255)
            
            # Invert the mask
            mask = 255 - mask[1:-1, 1:-1]
            
            # Option to feather edges
            if feather_edges:
                kernel_size = max(3, min(9, int(min(height, width) / 20)))
                mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
            
            # Apply alpha channel
            output[:, :, 3] = mask
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error removing background by edge: {str(e)}")
            # Return original image with full alpha
            rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image_array
            rgba[:, :, 3] = 255
            return rgba

    def save_transparent_image(self, image_array, output_path):
        """
        Save image with transparent background.
        
        Args:
            image_array: RGBA image array
            output_path: Path to save the transparent PNG
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create PIL image from RGBA array
            image = Image.fromarray(image_array)
            
            # Save as PNG (supports transparency)
            image.save(output_path, format='PNG')
            self.logger.info(f"Saved transparent image to {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving transparent image: {str(e)}")
            return False
