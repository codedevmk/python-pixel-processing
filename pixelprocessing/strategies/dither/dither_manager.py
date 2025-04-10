import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from skimage import color as skcolor
import logging
import math

class DitherManager:
    """
    Manages various dithering algorithms for pixel art processing.
    
    This class serves as a central coordinator for different dithering strategies,
    allowing users to apply various dithering algorithms to images.
    """
    
    def __init__(self, config):
        """
        Initialize the dither manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize strategies lazily
        self._floyd_steinberg = None
        self._bayer = None
        self._blue_noise = None
    
    @property
    def floyd_steinberg(self):
        """Lazy-load the Floyd-Steinberg dithering strategy."""
        if self._floyd_steinberg is None:
            from .floyd_steinberg import FloydSteinbergDither
            self._floyd_steinberg = FloydSteinbergDither(self.config)
        return self._floyd_steinberg
    
    @property
    def bayer(self):
        """Lazy-load the Bayer (ordered) dithering strategy."""
        if self._bayer is None:
            from .bayer import BayerDither
            self._bayer = BayerDither(self.config)
        return self._bayer
    
    @property
    def blue_noise(self):
        """Lazy-load the Blue Noise dithering strategy."""
        if self._blue_noise is None:
            from .blue_noise import BlueNoiseDither
            self._blue_noise = BlueNoiseDither(self.config)
        return self._blue_noise
    
    def apply_dithering(self, image_array, palette, dither_type='floyd-steinberg', strength=1.0):
        """
        Apply different dithering algorithms to an image.
        
        Args:
            image_array: Input image as numpy array (RGB)
            palette: Color palette to use as numpy array
            dither_type: Type of dithering ('none', 'floyd-steinberg', 'bayer', 'blue-noise')
            strength: Dithering strength from 0.0 to 1.0
            
        Returns:
            Dithered image as numpy array
        """
        self.logger.info(f"Applying {dither_type} dithering with strength {strength}")
        
        # Skip dithering if requested
        if dither_type.lower() == 'none':
            return self._map_to_palette_no_dither(image_array, palette)
            
        # Select appropriate dithering algorithm
        if dither_type.lower() == 'floyd-steinberg':
            return self.floyd_steinberg.apply_dither(image_array, palette, strength)
        elif dither_type.lower() == 'bayer':
            return self.bayer.apply_dither(image_array, palette, strength)
        elif dither_type.lower() == 'blue-noise':
            return self.blue_noise.apply_dither(image_array, palette, strength)
        else:
            self.logger.warning(f"Unknown dithering type: {dither_type}, falling back to floyd-steinberg")
            return self.floyd_steinberg.apply_dither(image_array, palette, strength)
    
    def _map_to_palette_no_dither(self, image_array, palette):
        """
        Map image colors to palette without dithering.
        
        Args:
            image_array: Input image as numpy array
            palette: Color palette as numpy array
            
        Returns:
            Image mapped to palette
        """
        try:
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
    
    def create_dither_comparison(self, image_array, palette, output_path):
        """
        Create a comparison image showing different dithering methods.
        
        Args:
            image_array: Input image as numpy array
            palette: Color palette to use
            output_path: Path to save the comparison image
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create three dithered versions
            no_dither = self._map_to_palette_no_dither(image_array, palette)
            floyd = self.floyd_steinberg.apply_dither(image_array, palette, 1.0)
            bayer = self.bayer.apply_dither(image_array, palette, 1.0)
            blue_noise = self.blue_noise.apply_dither(image_array, palette, 1.0)
            
            # Get dimensions
            height, width = image_array.shape[:2]
            
            # Create a combined image with all four versions
            combined = Image.new('RGB', (width * 2, height * 2))
            
            # Add labels for each algorithm
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined)
            
            # Try to load a font, use default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Paste the images
            combined.paste(Image.fromarray(no_dither.astype('uint8')), (0, 0))
            combined.paste(Image.fromarray(floyd.astype('uint8')), (width, 0))
            combined.paste(Image.fromarray(bayer.astype('uint8')), (0, height))
            combined.paste(Image.fromarray(blue_noise.astype('uint8')), (width, height))
            
            # Add labels
            draw.text((10, 10), "No Dithering", fill=(255, 255, 255), font=font)
            draw.text((width + 10, 10), "Floyd-Steinberg", fill=(255, 255, 255), font=font)
            draw.text((10, height + 10), "Bayer (Ordered)", fill=(255, 255, 255), font=font)
            draw.text((width + 10, height + 10), "Blue Noise", fill=(255, 255, 255), font=font)
            
            # Save the comparison image
            combined.save(output_path)
            self.logger.info(f"Saved dithering comparison to {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating dither comparison: {str(e)}")
            return False
