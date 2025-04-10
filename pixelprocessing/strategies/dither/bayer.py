import numpy as np
from skimage import color as skcolor
import logging
from scipy.spatial import KDTree
import time

class BayerDither:
    """
    Implements ordered (Bayer) dithering.
    
    Bayer dithering is a type of ordered dithering that uses a threshold map
    (Bayer matrix) to determine which pixels to adjust. It creates a regular
    pattern that is aesthetically pleasing for pixel art.
    """
    
    def __init__(self, config):
        """
        Initialize the Bayer dithering strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for Bayer matrices of different sizes
        self._bayer_matrices = {}
    
    def apply_dither(self, image_array, palette, strength=1.0):
        """
        Apply Bayer dithering to an image.
        
        Args:
            image_array: Input image as numpy array (RGB)
            palette: Color palette as numpy array
            strength: Dithering strength from 0.0 to 1.0
            
        Returns:
            Dithered image as numpy array
        """
        try:
            start_time = time.time()
            
            # Get matrix size from config (default to 8x8)
            matrix_size = self.config.get('bayer_matrix_size', 8)
            self.logger.info(f"Applying Bayer dithering with matrix size {matrix_size} and strength {strength}")
            
            height, width = image_array.shape[:2]
            
            # Generate Bayer matrix if not already cached
            if matrix_size not in self._bayer_matrices:
                self._bayer_matrices[matrix_size] = self._generate_bayer_matrix(matrix_size)
            
            bayer_matrix = self._bayer_matrices[matrix_size]
            
            # Normalize the matrix to range [-0.5, 0.5]
            normalized_matrix = (bayer_matrix / (matrix_size * matrix_size)) - 0.5
            
            # Scale matrix based on strength
            threshold_matrix = normalized_matrix * 30.0 * strength
            
            # Tile the threshold matrix to match image dimensions
            threshold_map = np.tile(
                threshold_matrix,
                (1 + height // matrix_size, 1 + width // matrix_size)
            )[:height, :width]
            
            # Convert image to LAB color space for better color perception
            img_lab = skcolor.rgb2lab(image_array / 255.0)
            palette_lab = skcolor.rgb2lab(palette / 255.0)
            
            # Create KDTree for efficient color mapping
            tree = KDTree(palette_lab)
            
            # Create output array
            output = np.zeros_like(image_array)
            
            # Apply threshold map to each pixel
            for y in range(height):
                for x in range(width):
                    # Get original pixel
                    pixel = img_lab[y, x].copy()
                    
                    # Apply threshold adjustment (only to L channel for better results)
                    adjusted_pixel = pixel.copy()
                    adjusted_pixel[0] += threshold_map[y, x]
                    
                    # Find nearest color in palette
                    idx = tree.query([adjusted_pixel], k=1)[1][0]
                    
                    # Store the mapped color in the output
                    output[y, x] = palette[idx]
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Bayer dithering completed in {elapsed_time:.2f} seconds")
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error applying Bayer dithering: {str(e)}")
            # Fall back to non-dithered version
            from .dither_manager import DitherManager
            return DitherManager(self.config)._map_to_palette_no_dither(image_array, palette)
    
    def _generate_bayer_matrix(self, size):
        """
        Generate a Bayer dithering matrix of the given size.
        
        Size should be a power of 2 (e.g., 2, 4, 8, 16).
        
        Args:
            size: Matrix size (power of 2)
            
        Returns:
            Generated Bayer matrix as numpy array
        """
        # Check if size is power of 2
        if not (size & (size - 1) == 0) or size <= 0:
            self.logger.warning(f"Bayer matrix size {size} is not a power of 2, using nearest power of 2")
            size = 1 << (size - 1).bit_length()  # Round to previous power of 2
        
        # Base 2x2 matrix
        bayer2x2 = np.array([[0, 2], [3, 1]])
        
        # If we just want the 2x2 matrix, return it
        if size <= 2:
            return bayer2x2
        
        # Otherwise, recursively expand to desired size
        n = 2
        bayer = bayer2x2
        
        while n < size:
            n *= 2
            # Expand using the recursive pattern
            bayer = np.block([
                [4 * bayer, 4 * bayer + 2],
                [4 * bayer + 3, 4 * bayer + 1]
            ])
        
        return bayer
    
    def apply_dither_optimized(self, image_array, palette, strength=1.0):
        """
        Apply Bayer dithering with optimizations for larger images.
        
        This version uses vectorized operations for better performance.
        
        Args:
            image_array: Input image as numpy array (RGB)
            palette: Color palette as numpy array
            strength: Dithering strength from 0.0 to 1.0
            
        Returns:
            Dithered image as numpy array
        """
        try:
            start_time = time.time()
            
            # Get matrix size from config (default to 8x8)
            matrix_size = self.config.get('bayer_matrix_size', 8)
            self.logger.info(f"Applying optimized Bayer dithering with matrix size {matrix_size}")
            
            height, width = image_array.shape[:2]
            
            # Generate Bayer matrix if not already cached
            if matrix_size not in self._bayer_matrices:
                self._bayer_matrices[matrix_size] = self._generate_bayer_matrix(matrix_size)
            
            bayer_matrix = self._bayer_matrices[matrix_size]
            
            # Normalize the matrix to range [-0.5, 0.5]
            normalized_matrix = (bayer_matrix / (matrix_size * matrix_size)) - 0.5
            
            # Scale matrix based on strength
            threshold_matrix = normalized_matrix * 30.0 * strength
            
            # Tile the threshold matrix to match image dimensions
            threshold_map = np.tile(
                threshold_matrix,
                (1 + height // matrix_size, 1 + width // matrix_size)
            )[:height, :width]
            
            # Convert to LAB color space
            img_lab = skcolor.rgb2lab(image_array / 255.0)
            
            # Apply adjustment to L channel only (first channel in LAB)
            adjusted_lab = img_lab.copy()
            adjusted_lab[:, :, 0] += threshold_map
            
            # Convert back to RGB for output
            adjusted_rgb = skcolor.lab2rgb(adjusted_lab) * 255.0
            
            # Apply palette mapping
            from .dither_manager import DitherManager
            result = DitherManager(self.config)._map_to_palette_no_dither(adjusted_rgb.astype(np.uint8), palette)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Optimized Bayer dithering completed in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying optimized Bayer dithering: {str(e)}")
            # Fall back to non-dithered version
            from .dither_manager import DitherManager
            return DitherManager(self.config)._map_to_palette_no_dither(image_array, palette)
