import os
import numpy as np
from PIL import Image
import logging

from .core.config import Config

class PixelProcessor:
    """
    Main class for processing pixel art images.
    
    This class provides a high-level interface to the various image processing
    capabilities of the library, including palette extraction, dithering,
    background removal, and special effects.
    """
    
    def __init__(self, image_path=None, image=None):
        """
        Initialize the PixelProcessor with either a file path or a PIL image.
        
        Args:
            image_path (str): Path to the image file to process.
            image (PIL.Image): A PIL Image object to process.
        
        Raises:
            ValueError: If neither image_path nor image is provided.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = Config()
        
        if image_path is not None:
            self.image_path = image_path
            self.image = Image.open(image_path)
        elif image is not None:
            self.image_path = None
            self.image = image
        else:
            raise ValueError("Either image_path or image must be provided")
        
        # Convert image to numpy array
        self.image_array = np.array(self.image)
        
        # Initialize processing components lazily
        self._palette_extractor = None
        self._color_mapper = None
        self._dither_manager = None
        self._background_remover = None
    
    @property
    def palette_extractor(self):
        """Lazy-load the palette extractor."""
        if self._palette_extractor is None:
            from .strategies.color.palette_extractor import PaletteExtractor
            self._palette_extractor = PaletteExtractor(self.config)
        return self._palette_extractor
    
    @property
    def color_mapper(self):
        """Lazy-load the color mapper."""
        if self._color_mapper is None:
            from .strategies.color.color_mapper import ColorMapper
            self._color_mapper = ColorMapper(self.config)
        return self._color_mapper
    
    @property
    def dither_manager(self):
        """Lazy-load the dither manager."""
        if self._dither_manager is None:
            from .strategies.dither.dither_manager import DitherManager
            self._dither_manager = DitherManager(self.config)
        return self._dither_manager
    
    @property
    def background_remover(self):
        """Lazy-load the background remover."""
        if self._background_remover is None:
            from .strategies.background.background_remover import BackgroundRemover
            self._background_remover = BackgroundRemover(self.config)
        return self._background_remover
    
    def extract_palette(self, colors=8, method='kmeans'):
        """
        Extract a color palette from the image.
        
        Args:
            colors (int): Number of colors to extract.
            method (str): Method to use for extraction ('kmeans' or 'median-cut').
        
        Returns:
            numpy.ndarray: Array of RGB colors forming the palette.
        """
        self.config.num_dominant_colors = colors
        
        if method == 'kmeans':
            return self.palette_extractor.get_dominant_colors(self.image_array)
        else:
            # For now, default to kmeans for other methods
            self.logger.warning(f"Method {method} not supported, using kmeans")
            return self.palette_extractor.get_dominant_colors(self.image_array)
    
    def apply_palette(self, palette, dither=None, dither_strength=1.0):
        """
        Apply a color palette to the image, optionally with dithering.
        
        Args:
            palette (numpy.ndarray): Array of RGB colors to use.
            dither (str): Dithering method ('none', 'floyd-steinberg', 'bayer', 'blue-noise').
            dither_strength (float): Strength of dithering effect (0.0-1.0).
        
        Returns:
            PIL.Image: Processed image.
        """
        if dither and dither.lower() != 'none':
            result_array = self.dither_manager.apply_dithering(
                self.image_array, palette, dither_type=dither, strength=dither_strength
            )
        else:
            result_array = self.color_mapper.map_image_to_palette(self.image_array, palette)
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def remove_background(self, method='color', threshold=30, feather_edges=False):
        """
        Remove the background from the image.
        
        Args:
            method (str): Method to use ('color', 'contrast', 'edge').
            threshold (int): Threshold for background detection.
            feather_edges (bool): Whether to apply soft edges.
        
        Returns:
            PIL.Image: RGBA image with transparent background.
        """
        transparent_array = self.background_remover.remove_background(
            self.image_array,
            method=method,
            color_threshold=threshold,
            feather_edges=feather_edges
        )
        
        return Image.fromarray(transparent_array)
    
    def resize(self, width=None, height=None, scale=None, method='pixel-perfect'):
        """
        Resize the image using various methods optimized for pixel art.
        
        Args:
            width (int): Target width in pixels.
            height (int): Target height in pixels.
            scale (float): Scale factor (e.g., 2.0 for 2x size).
            method (str): Resize method ('pixel-perfect', 'nearest', 'lanczos').
        
        Returns:
            PIL.Image: Resized image.
        """
        # Calculate dimensions
        orig_width, orig_height = self.image.size
        
        if scale is not None:
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
        else:
            # If width or height is provided, maintain aspect ratio
            if width is not None and height is None:
                scale = width / orig_width
                new_width = width
                new_height = int(orig_height * scale)
            elif height is not None and width is None:
                scale = height / orig_height
                new_height = height
                new_width = int(orig_width * scale)
            elif width is not None and height is not None:
                new_width = width
                new_height = height
            else:
                # No dimensions provided, return original
                return self.image
        
        # Choose resize method
        if method == 'pixel-perfect':
            # For pixel-perfect, ensure integer scaling
            if scale is not None and scale == int(scale):
                # Use nearest neighbor for integer scales
                resampling = Image.NEAREST
            else:
                self.logger.warning("Pixel-perfect scaling requires integer scale. Using nearest neighbor.")
                resampling = Image.NEAREST
        elif method == 'nearest':
            resampling = Image.NEAREST
        elif method == 'lanczos':
            resampling = Image.LANCZOS
        else:
            self.logger.warning(f"Unknown resize method: {method}, using nearest neighbor")
            resampling = Image.NEAREST
        
        return self.image.resize((new_width, new_height), resampling)
    
    def apply_crt_effect(self, scanline_strength=0.2, curvature=0.1):
        """
        Apply a retro CRT screen effect to the image.
        
        Args:
            scanline_strength (float): Strength of scanline effect (0.0-1.0).
            curvature (float): Amount of screen curvature (0.0-0.5).
        
        Returns:
            PIL.Image: Image with CRT effect.
        """
        from .effects.crt_effect import apply_crt_effect
        
        result_array = apply_crt_effect(
            self.image_array,
            scanline_strength=scanline_strength,
            curvature=curvature
        )
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def enhance_pixel_art(self, outline_color=None, thickness=1):
        """
        Enhance pixel art by detecting and improving outlines.
        
        Args:
            outline_color (tuple): RGB color for outlines (if None, uses darkest color).
            thickness (int): Outline thickness in pixels (1 or 2).
        
        Returns:
            PIL.Image: Enhanced image.
        """
        from .effects.pixel_art_effects import enhance_outlines
        
        result_array = enhance_outlines(
            self.image_array,
            outline_color=outline_color,
            thickness=thickness
        )
        
        return Image.fromarray(result_array.astype(np.uint8))
    
    def create_palette_cycling_animation(self, palette, cycle_range=(0, 8), frames=12, duration=100):
        """
        Create an animated GIF with palette cycling effects.
        
        Args:
            palette (numpy.ndarray): Color palette to use.
            cycle_range (tuple): Range of palette indices to cycle (start, end).
            frames (int): Number of frames in the animation.
            duration (int): Duration of each frame in milliseconds.
        
        Returns:
            PIL.Image: Animated GIF with palette cycling.
        """
        from .effects.animation_effects import create_palette_cycling
        
        # First map the image to the palette
        palette_mapped = self.color_mapper.map_image_to_palette(self.image_array, palette)
        
        # Create the animation frames
        animation_frames = create_palette_cycling(
            palette_mapped,
            palette,
            cycle_range=cycle_range,
            frames=frames
        )
        
        # Convert to PIL images
        pil_frames = [Image.fromarray(frame) for frame in animation_frames]
        
        # Create animated GIF
        animated = pil_frames[0].copy()
        animated.save(
            "temp_animation.gif",
            save_all=True,
            append_images=pil_frames[1:],
            optimize=False,
            duration=duration,
            loop=0
        )
        
        # Read the saved GIF back
        animated_gif = Image.open("temp_animation.gif")
        os.remove("temp_animation.gif")
        
        return animated_gif
    
    def save(self, output_path, format=None, **kwargs):
        """
        Save the current image to disk.
        
        Args:
            output_path (str): Path to save the image.
            format (str): Image format (will be inferred from extension if None).
            **kwargs: Additional arguments to pass to PIL.Image.save().
        
        Returns:
            bool: True if successful.
        """
        try:
            self.image.save(output_path, format=format, **kwargs)
            self.logger.info(f"Image saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving image: {str(e)}")
            return False
