import numpy as np
from scipy.spatial import KDTree
from skimage import color as skcolor
import logging

class ColorMapper:
    """
    Handles mapping colors between images and palettes.
    
    This class provides methods to efficiently map images to color palettes
    using perceptual color spaces and nearest-neighbor algorithms.
    """
    
    def __init__(self, config):
        """
        Initialize the color mapper.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def map_image_to_palette(self, image_array, palette):
        """
        Remap each pixel in the image to the nearest palette color using LAB color space.
        
        Args:
            image_array: Input image as numpy array
            palette: Color palette as numpy array
            
        Returns:
            Mapped image as numpy array
        """
        try:
            # Make sure we have valid data
            if image_array.size == 0:
                raise ValueError("Empty image array")
            if palette.size == 0:
                raise ValueError("Empty palette")
            
            # Reshape image for processing
            original_shape = image_array.shape
            flat_image = image_array.reshape(-1, 3)
            
            # Convert to LAB color space for perceptual color mapping
            image_lab = skcolor.rgb2lab(flat_image / 255.0)
            palette_lab = skcolor.rgb2lab(palette / 255.0)
            
            # Use KDTree for efficient nearest neighbor lookup
            tree = KDTree(palette_lab)
            indices = tree.query(image_lab, k=1)[1]
            
            # Map to palette colors and reshape to original dimensions
            mapped_colors = palette[indices]
            mapped_image = mapped_colors.reshape(original_shape)
            
            return mapped_image
            
        except Exception as e:
            self.logger.error(f"Error mapping image to palette: {str(e)}")
            # Return the original image if mapping fails
            return image_array
        
    def analyze_and_correct_palette(self, dominant_colors, mapped_colors, palette):
        """
        Adjust palette if dominant color deviates too much from mapped color.
        
        This method helps preserve important colors in the original image
        by adjusting the palette to better match the dominant colors.
        
        Args:
            dominant_colors: Dominant colors from the original image
            mapped_colors: How those colors map to the current palette
            palette: The palette to adjust
            
        Returns:
            Corrected palette as numpy array
        """
        try:
            corrected_palette = palette.copy()
            palette_rgb_to_index = {tuple(c): i for i, c in enumerate(corrected_palette)}
            
            for original, mapped in zip(dominant_colors, mapped_colors):
                # Calculate color distance
                dist = np.linalg.norm(np.array(original) - np.array(mapped))
                
                # If the distance exceeds the threshold, adjust the palette
                if dist > self.config.color_distance_threshold:
                    # Find the mapped color in the palette
                    mapped_idx = palette_rgb_to_index.get(tuple(mapped))
                    
                    # Replace it with the original color
                    if mapped_idx is not None:
                        corrected_palette[mapped_idx] = original
                        self.logger.debug(f"Replaced palette color {mapped} with {original}")
            
            return corrected_palette
            
        except Exception as e:
            self.logger.error(f"Error correcting palette: {str(e)}")
            # Return the original palette if correction fails
            return palette
    
    def map_image_to_palette_dithered(self, image_array, palette, threshold_map=None):
        """
        Map an image to a palette with threshold map-based dithering.
        
        Args:
            image_array: Input image as numpy array
            palette: Color palette as numpy array
            threshold_map: Threshold map for dithering
            
        Returns:
            Dithered image as numpy array
        """
        try:
            height, width = image_array.shape[:2]
            result = np.zeros_like(image_array)
            
            # Use threshold map if provided
            if threshold_map is not None:
                # Ensure threshold map matches image dimensions
                if threshold_map.shape[:2] != (height, width):
                    # Tile the threshold map to match image dimensions
                    th_height, th_width = threshold_map.shape
                    tiled_map = np.tile(
                        threshold_map, 
                        (1 + height // th_height, 1 + width // th_width)
                    )
                    threshold_map = tiled_map[:height, :width]
                
                # Apply threshold adjustment
                adjusted_image = image_array.astype(float)
                
                # Apply threshold map to each channel
                for c in range(3):
                    adjusted_image[:, :, c] += threshold_map * self.config.get('dither_strength', 1.0)
                
                # Clip values to valid range
                adjusted_image = np.clip(adjusted_image, 0, 255)
                
                # Map the adjusted image to the palette
                result = self.map_image_to_palette(adjusted_image.astype(np.uint8), palette)
                
            else:
                # If no threshold map, just do regular mapping
                result = self.map_image_to_palette(image_array, palette)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying dithered mapping: {str(e)}")
            # Return regular mapping if dithering fails
            return self.map_image_to_palette(image_array, palette)
