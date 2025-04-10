import numpy as np
from skimage import color as skcolor
import logging
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import time
import os

class BlueNoiseDither:
    """
    Implements blue noise dithering.
    
    Blue noise dithering uses a blue noise pattern for threshold mapping,
    which creates a more organic, natural-looking dither pattern compared
    to ordered dithering, while avoiding the directional artifacts of
    error diffusion methods.
    """
    
    def __init__(self, config):
        """
        Initialize the Blue Noise dithering strategy.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Blue noise pattern cache
        self._blue_noise_patterns = {}
        
        # Create patterns directory if it doesn't exist
        self.patterns_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'patterns'
        )
        os.makedirs(self.patterns_dir, exist_ok=True)
    
    def apply_dither(self, image_array, palette, strength=1.0):
        """
        Apply blue noise dithering to an image.
        
        Args:
            image_array: Input image as numpy array (RGB)
            palette: Color palette as numpy array
            strength: Dithering strength from 0.0 to 1.0
            
        Returns:
            Dithered image as numpy array
        """
        try:
            start_time = time.time()
            
            # Get pattern size from config (default to 64x64)
            pattern_size = self.config.get('blue_noise_pattern_size', 64)
            self.logger.info(f"Applying blue noise dithering with pattern size {pattern_size} and strength {strength}")
            
            height, width = image_array.shape[:2]
            
            # Get or generate blue noise pattern
            pattern = self._get_blue_noise_pattern(pattern_size)
            
            # Normalize pattern to range [-0.5, 0.5]
            normalized_pattern = (pattern / 255.0) - 0.5
            
            # Scale pattern based on strength and adjust for image brightness
            threshold_matrix = normalized_pattern * 30.0 * strength
            
            # Tile the pattern to match image dimensions
            threshold_map = np.tile(
                threshold_matrix,
                (1 + height // pattern_size, 1 + width // pattern_size)
            )[:height, :width]
            
            # Convert to LAB color space for better color perception
            img_lab = skcolor.rgb2lab(image_array / 255.0)
            palette_lab = skcolor.rgb2lab(palette / 255.0)
            
            # Create KDTree for efficient color mapping
            tree = KDTree(palette_lab)
            
            # Create output array
            output = np.zeros_like(image_array)
            
            # Apply threshold map to each pixel (using luminance channel)
            for y in range(height):
                for x in range(width):
                    # Get original pixel
                    pixel = img_lab[y, x].copy()
                    
                    # Apply threshold adjustment (only to L channel)
                    adjusted_pixel = pixel.copy()
                    adjusted_pixel[0] += threshold_map[y, x]
                    
                    # Find nearest color in palette
                    idx = tree.query([adjusted_pixel], k=1)[1][0]
                    
                    # Store the mapped color in the output
                    output[y, x] = palette[idx]
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Blue noise dithering completed in {elapsed_time:.2f} seconds")
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error applying blue noise dithering: {str(e)}")
            # Fall back to non-dithered version
            from .dither_manager import DitherManager
            return DitherManager(self.config)._map_to_palette_no_dither(image_array, palette)
    
    def _get_blue_noise_pattern(self, size):
        """
        Get a blue noise pattern of the specified size.
        
        Tries to load from cache or disk first, generates if not found.
        
        Args:
            size: Desired pattern size
            
        Returns:
            Blue noise pattern as numpy array
        """
        # Check if we have it cached in memory
        if size in self._blue_noise_patterns:
            self.logger.debug(f"Using cached {size}x{size} blue noise pattern")
            return self._blue_noise_patterns[size]
        
        # Check if we have a pre-generated pattern file
        pattern_file = os.path.join(self.patterns_dir, f"blue_noise_{size}x{size}.npy")
        
        if os.path.exists(pattern_file):
            try:
                self.logger.info(f"Loading blue noise pattern from {pattern_file}")
                pattern = np.load(pattern_file)
                
                # Cache it for future use
                self._blue_noise_patterns[size] = pattern
                return pattern
            except Exception as e:
                self.logger.warning(f"Failed to load blue noise pattern from file: {e}")
        
        # Generate a new pattern
        self.logger.info(f"Generating {size}x{size} blue noise pattern")
        pattern = self._generate_blue_noise(size, size)
        
        # Cache it for future use
        self._blue_noise_patterns[size] = pattern
        
        # Save it to disk for future runs
        try:
            np.save(pattern_file, pattern)
            self.logger.info(f"Saved blue noise pattern to {pattern_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save blue noise pattern: {e}")
        
        return pattern
    
    def _generate_blue_noise(self, width, height):
        """
        Generate a blue noise pattern.
        
        This simplified implementation uses a combination of white noise
        and filtering techniques to create a blue noise-like pattern.
        
        Args:
            width: Pattern width
            height: Pattern height
            
        Returns:
            Blue noise pattern as numpy array
        """
        self.logger.info(f"Generating {width}x{height} blue noise pattern")
        
        # For this simplified implementation, generate white noise and filter it
        # A more accurate blue noise generator would use void-and-cluster algorithm
        
        # Generate white noise
        np.random.seed(42)  # For reproducibility
        white_noise = np.random.rand(height, width) - 0.5  # Range [-0.5, 0.5]
        
        # Apply filters to shape frequency distribution to be more blue-noise-like
        sigma_small = 0.5
        sigma_large = 1.0
        filtered = gaussian_filter(white_noise, sigma_large) - gaussian_filter(white_noise, sigma_small)
        
        # Normalize to range [0, 255]
        filtered_min = filtered.min()
        filtered_max = filtered.max()
        blue_noise = ((filtered - filtered_min) / (filtered_max - filtered_min) * 255).astype(np.uint8)
        
        # Apply histogram equalization to ensure good distribution
        hist, bins = np.histogram(blue_noise, 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        equalized = np.interp(blue_noise.flatten(), bins[:-1], cdf_normalized).reshape(blue_noise.shape)
        
        return equalized.astype(np.uint8)
    
    def apply_dither_optimized(self, image_array, palette, strength=1.0):
        """
        Apply blue noise dithering with optimization for larger images.
        
        Uses vectorized operations for better performance.
        
        Args:
            image_array: Input image as numpy array (RGB)
            palette: Color palette as numpy array
            strength: Dithering strength from 0.0 to 1.0
            
        Returns:
            Dithered image as numpy array
        """
        try:
            start_time = time.time()
            self.logger.info(f"Applying optimized blue noise dithering with strength {strength}")
            
            height, width = image_array.shape[:2]
            
            # Get or generate blue noise pattern
            pattern_size = self.config.get('blue_noise_pattern_size', 64)
            pattern = self._get_blue_noise_pattern(pattern_size)
            
            # Tile the pattern to match image dimensions
            tiled_pattern = np.tile(
                pattern, 
                (1 + height // pattern_size, 1 + width // pattern_size)
            )[:height, :width]
            
            # Normalize and scale the pattern
            normalized_pattern = (tiled_pattern / 255.0 - 0.5) * strength
            
            # Convert image to LAB color space
            img_lab = skcolor.rgb2lab(image_array / 255.0)
            
            # Apply threshold adjustment to L channel only
            adjusted_lab = img_lab.copy()
            adjusted_lab[:, :, 0] += normalized_pattern * 30.0
            
            # Convert back to RGB
            adjusted_rgb = skcolor.lab2rgb(adjusted_lab) * 255.0
            
            # Apply palette mapping
            from .dither_manager import DitherManager
            result = DitherManager(self.config)._map_to_palette_no_dither(
                adjusted_rgb.astype(np.uint8), palette
            )
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Optimized blue noise dithering completed in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying optimized blue noise dithering: {str(e)}")
            # Fall back to non-dithered version
            from .dither_manager import DitherManager
            return DitherManager(self.config)._map_to_palette_no_dither(image_array, palette)
