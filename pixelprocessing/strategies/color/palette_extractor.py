import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import logging
import os

class PaletteExtractor:
    """
    Handles palette extraction and color analysis.
    
    This class provides methods to extract color palettes from images using
    various techniques such as k-means clustering, median cut, and octree.
    """
    
    def __init__(self, config):
        """
        Initialize the palette extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract_palette(self, palette_path):
        """
        Extract unique colors from a palette image.
        
        Args:
            palette_path: Path to the palette image
            
        Returns:
            Numpy array of unique colors
        """
        try:
            img = Image.open(palette_path).convert("RGB")
            pixels = np.array(img).reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            
            if len(unique_colors) == 0:
                raise ValueError(f"No colors found in palette image: {palette_path}")
            
            self.logger.info(f"Extracted {len(unique_colors)} unique colors from palette image")
            return unique_colors
            
        except Exception as e:
            self.logger.error(f"Error extracting palette from {palette_path}: {str(e)}")
            raise
    
    def save_palette_image(self, palette, output_path, width=600, height=50):
        """
        Save a palette as an image for reference.
        
        Args:
            palette: Numpy array of RGB colors
            output_path: Path to save the palette image
            width: Width of the output image
            height: Height of each color stripe
            
        Returns:
            Boolean indicating success
        """
        try:
            palette_size = len(palette)
            if palette_size == 0:
                raise ValueError("Cannot save empty palette")
                
            # Create a new image with white background
            palette_img = Image.new('RGB', (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(palette_img)
            
            # Draw each color as a rectangle
            width_per_color = width / palette_size
            for i, color in enumerate(palette):
                x0 = i * width_per_color
                x1 = (i + 1) * width_per_color
                draw.rectangle([x0, 0, x1, height], fill=tuple(color))
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the palette image
            palette_img.save(output_path)
            self.logger.info(f"Saved palette with {palette_size} colors to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save palette image: {str(e)}")
            return False
        
    def get_dominant_colors(self, image_array):
        """
        Extract dominant colors using KMeans clustering.
        
        Args:
            image_array: Input image as numpy array
            
        Returns:
            Numpy array of dominant RGB colors
        """
        try:
            # Reshape the image to a list of pixels
            pixels = image_array.reshape(-1, 3)
            
            # Handle case of very small images or uniform colors
            n_colors = self.config.num_dominant_colors
            if pixels.shape[0] < n_colors:
                self.logger.warning(f"Image has fewer pixels ({pixels.shape[0]}) than requested colors ({n_colors})")
                n_colors = max(1, pixels.shape[0])
            
            # For larger images, sample pixels to improve performance
            if pixels.shape[0] > 10000:
                self.logger.info(f"Sampling {10000} pixels from {pixels.shape[0]} total pixels")
                indices = np.random.choice(pixels.shape[0], 10000, replace=False)
                sample_pixels = pixels[indices]
            else:
                sample_pixels = pixels
            
            # Use KMeans to find dominant colors
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(sample_pixels)
            centers = kmeans.cluster_centers_
            
            # Convert to uint8
            centers = centers.astype(int)
            
            self.logger.info(f"Extracted {n_colors} dominant colors")
            return centers
            
        except Exception as e:
            self.logger.error(f"Error extracting dominant colors: {str(e)}")
            # Fallback to a simple approach if KMeans fails
            return np.array([[0, 0, 0], [255, 255, 255]])
    
    def extract_palette_median_cut(self, image_array, n_colors=8):
        """
        Extract a color palette using the median cut algorithm.
        
        Args:
            image_array: Input image as numpy array
            n_colors: Number of colors to extract
            
        Returns:
            Numpy array of RGB colors
        """
        try:
            # Reshape the image to a list of pixels
            pixels = image_array.reshape(-1, 3)
            
            # Recursive median cut
            colors = self._median_cut(pixels, n_colors)
            
            self.logger.info(f"Extracted {len(colors)} colors using median cut")
            return np.array(colors)
            
        except Exception as e:
            self.logger.error(f"Error extracting palette with median cut: {str(e)}")
            # Fallback to KMeans
            return self.get_dominant_colors(image_array)
    
    def _median_cut(self, pixels, depth):
        """
        Recursive helper for median cut algorithm.
        
        Args:
            pixels: Array of pixels
            depth: Remaining recursion depth
            
        Returns:
            List of RGB colors
        """
        if depth == 0 or len(pixels) == 0:
            # Calculate average color in this bucket
            avg_color = np.mean(pixels, axis=0).astype(int)
            return [avg_color]
        
        # Find channel with greatest range
        r_range = np.max(pixels[:, 0]) - np.min(pixels[:, 0])
        g_range = np.max(pixels[:, 1]) - np.min(pixels[:, 1])
        b_range = np.max(pixels[:, 2]) - np.min(pixels[:, 2])
        
        channel = np.argmax([r_range, g_range, b_range])
        
        # Sort pixels by the selected channel
        pixels = pixels[pixels[:, channel].argsort()]
        
        # Split the pixels at the median
        median = len(pixels) // 2
        left = pixels[:median]
        right = pixels[median:]
        
        # Recurse on both halves
        return (self._median_cut(left, depth - 1) + 
                self._median_cut(right, depth - 1))
    
    def quantize_image(self, image_array, n_colors=8, method='kmeans'):
        """
        Quantize an image to a specific number of colors.
        
        Args:
            image_array: Input image as numpy array
            n_colors: Number of colors to use
            method: Quantization method ('kmeans' or 'median-cut')
            
        Returns:
            Tuple of (quantized image, palette)
        """
        try:
            # Extract the palette based on the specified method
            if method == 'kmeans':
                self.config.num_dominant_colors = n_colors
                palette = self.get_dominant_colors(image_array)
            elif method == 'median-cut':
                palette = self.extract_palette_median_cut(image_array, n_colors)
            else:
                self.logger.warning(f"Unknown method: {method}, using kmeans")
                self.config.num_dominant_colors = n_colors
                palette = self.get_dominant_colors(image_array)
            
            # Map the image to the palette
            from .color_mapper import ColorMapper
            mapper = ColorMapper(self.config)
            quantized = mapper.map_image_to_palette(image_array, palette)
            
            return quantized, palette
            
        except Exception as e:
            self.logger.error(f"Error quantizing image: {str(e)}")
            # Return original image and a basic palette
            return image_array, np.array([[0, 0, 0], [255, 255, 255]])
