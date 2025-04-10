import numpy as np
import time
from skimage import color as skcolor
import logging
from scipy.spatial import KDTree

class FloydSteinbergDither:
    """
    Implements Floyd-Steinberg error-diffusion dithering.
    
    Floyd-Steinberg dithering is a popular error-diffusion algorithm that
    propagates quantization errors to neighboring pixels, resulting in a
    more natural-looking dithered image compared to ordered dithering.
    """
    
    def __init__(self, config):
        """
        Initialize the Floyd-Steinberg dithering strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default error diffusion coefficients
        self.diffusion_matrix = {
            'right': 7/16,
            'bottom_left': 3/16,
            'bottom': 5/16,
            'bottom_right': 1/16
        }
    
    def apply_dither(self, image_array, palette, strength=1.0):
        """
        Apply Floyd-Steinberg dithering to an image.
        
        Args:
            image_array: Input image as numpy array (RGB)
            palette: Color palette as numpy array
            strength: Dithering strength from 0.0 to 1.0
            
        Returns:
            Dithered image as numpy array
        """
        try:
            start_time = time.time()
            self.logger.info(f"Applying Floyd-Steinberg dithering with strength {strength}")
            
            height, width = image_array.shape[:2]
            
            # Create a copy of the image in float format for error diffusion
            img_lab = skcolor.rgb2lab(image_array / 255.0).astype(np.float64)
            
            # Convert palette to LAB color space
            palette_lab = skcolor.rgb2lab(palette / 255.0)
            
            # Create output array
            output = np.zeros_like(image_array)
            
            # Create KDTree for efficient color mapping
            tree = KDTree(palette_lab)
            
            # Define diffusion matrix (can be customized via config)
            diffusion = self.config.get('diffusion_matrix', self.diffusion_matrix)
            
            # Adjust strength for the error diffusion
            right_coeff = diffusion['right'] * strength
            bottom_left_coeff = diffusion['bottom_left'] * strength
            bottom_coeff = diffusion['bottom'] * strength
            bottom_right_coeff = diffusion['bottom_right'] * strength
            
            # Process each pixel in sequence
            for y in range(height):
                for x in range(width):
                    # Get current pixel with accumulated error
                    old_pixel = img_lab[y, x].copy()
                    
                    # Find nearest color in palette
                    idx = tree.query([old_pixel], k=1)[1][0]
                    new_pixel = palette_lab[idx]
                    
                    # Store the mapped color in the output
                    output[y, x] = palette[idx]
                    
                    # Calculate quantization error
                    quant_error = old_pixel - new_pixel
                    
                    # Diffuse error to neighboring pixels
                    if x + 1 < width:
                        img_lab[y, x + 1] += quant_error * right_coeff
                    
                    if y + 1 < height:
                        if x > 0:
                            img_lab[y + 1, x - 1] += quant_error * bottom_left_coeff
                        
                        img_lab[y + 1, x] += quant_error * bottom_coeff
                        
                        if x + 1 < width:
                            img_lab[y + 1, x + 1] += quant_error * bottom_right_coeff
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Floyd-Steinberg dithering completed in {elapsed_time:.2f} seconds")
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error applying Floyd-Steinberg dithering: {str(e)}")
            # Fall back to non-dithered version
            from .dither_manager import DitherManager
            return DitherManager(self.config)._map_to_palette_no_dither(image_array, palette)
    
    def apply_serpentine_dither(self, image_array, palette, strength=1.0):
        """
        Apply serpentine Floyd-Steinberg dithering to an image.
        
        This variation alternates the scanning direction for each row,
        which helps reduce pattern artifacts.
        
        Args:
            image_array: Input image as numpy array (RGB)
            palette: Color palette as numpy array
            strength: Dithering strength from 0.0 to 1.0
            
        Returns:
            Dithered image as numpy array
        """
        try:
            start_time = time.time()
            self.logger.info(f"Applying serpentine Floyd-Steinberg dithering with strength {strength}")
            
            height, width = image_array.shape[:2]
            
            # Create a copy of the image in float format for error diffusion
            img_lab = skcolor.rgb2lab(image_array / 255.0).astype(np.float64)
            
            # Convert palette to LAB color space
            palette_lab = skcolor.rgb2lab(palette / 255.0)
            
            # Create output array
            output = np.zeros_like(image_array)
            
            # Create KDTree for efficient color mapping
            tree = KDTree(palette_lab)
            
            # Define diffusion matrix (can be customized via config)
            diffusion = self.config.get('diffusion_matrix', self.diffusion_matrix)
            
            # Adjust strength for the error diffusion
            right_coeff = diffusion['right'] * strength
            bottom_left_coeff = diffusion['bottom_left'] * strength
            bottom_coeff = diffusion['bottom'] * strength
            bottom_right_coeff = diffusion['bottom_right'] * strength
            
            # Process each pixel in sequence, with alternating row directions
            for y in range(height):
                # Determine direction for this row
                left_to_right = (y % 2 == 0)
                
                # Create row range based on direction
                if left_to_right:
                    x_range = range(width)
                else:
                    x_range = range(width - 1, -1, -1)
                
                for x in x_range:
                    # Get current pixel with accumulated error
                    old_pixel = img_lab[y, x].copy()
                    
                    # Find nearest color in palette
                    idx = tree.query([old_pixel], k=1)[1][0]
                    new_pixel = palette_lab[idx]
                    
                    # Store the mapped color in the output
                    output[y, x] = palette[idx]
                    
                    # Calculate quantization error
                    quant_error = old_pixel - new_pixel
                    
                    # Diffuse error to neighboring pixels based on direction
                    if left_to_right:
                        if x + 1 < width:
                            img_lab[y, x + 1] += quant_error * right_coeff
                        
                        if y + 1 < height:
                            if x > 0:
                                img_lab[y + 1, x - 1] += quant_error * bottom_left_coeff
                            
                            img_lab[y + 1, x] += quant_error * bottom_coeff
                            
                            if x + 1 < width:
                                img_lab[y + 1, x + 1] += quant_error * bottom_right_coeff
                    else:
                        # Right to left diffusion
                        if x > 0:
                            img_lab[y, x - 1] += quant_error * right_coeff
                        
                        if y + 1 < height:
                            if x + 1 < width:
                                img_lab[y + 1, x + 1] += quant_error * bottom_left_coeff
                            
                            img_lab[y + 1, x] += quant_error * bottom_coeff
                            
                            if x > 0:
                                img_lab[y + 1, x - 1] += quant_error * bottom_right_coeff
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Serpentine Floyd-Steinberg dithering completed in {elapsed_time:.2f} seconds")
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error applying serpentine Floyd-Steinberg dithering: {str(e)}")
            # Fall back to regular Floyd-Steinberg
            return self.apply_dither(image_array, palette, strength)
