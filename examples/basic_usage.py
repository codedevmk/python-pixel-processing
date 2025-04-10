#!/usr/bin/env python3
"""
Basic usage examples for the pixelprocessing library.

This script demonstrates how to use the PixelProcessor class to perform
various image processing operations.
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np

# Add parent directory to path to import pixelprocessing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pixelprocessing import PixelProcessor

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def example_extract_palette(input_image, output_dir):
    """Example of extracting a color palette from an image."""
    print("\n=== Extracting Color Palette ===")
    
    processor = PixelProcessor(input_image)
    
    # Extract a palette with 8 colors
    palette = processor.extract_palette(colors=8)
    print(f"Extracted {len(palette)} colors from {input_image}")
    
    # Save palette image
    palette_path = os.path.join(output_dir, "palette.png")
    
    # Use PaletteExtractor directly to save palette image
    from pixelprocessing.strategies.color.palette_extractor import PaletteExtractor
    from pixelprocessing.core.config import Config
    
    palette_extractor = PaletteExtractor(Config())
    palette_extractor.save_palette_image(palette, palette_path)
    
    print(f"Saved palette to {palette_path}")
    return palette

def example_apply_dithering(input_image, output_dir, palette):
    """Example of applying various dithering methods."""
    print("\n=== Applying Dithering ===")
    
    processor = PixelProcessor(input_image)
    
    # Apply different dithering methods
    methods = ['none', 'floyd-steinberg', 'bayer', 'blue-noise']
    
    for method in methods:
        print(f"Applying {method} dithering...")
        result = processor.apply_palette(palette, method)
        
        # Save result
        output_path = os.path.join(output_dir, f"dithered_{method}.png")
        result.save(output_path)
        print(f"Saved to {output_path}")
    
    # Create a dithering comparison
    from pixelprocessing.strategies.dither.dither_manager import DitherManager
    from pixelprocessing.core.config import Config
    
    dither_manager = DitherManager(Config())
    comparison_path = os.path.join(output_dir, "dither_comparison.png")
    
    print("Creating dithering comparison...")
    dither_manager.create_dither_comparison(
        np.array(processor.image),
        palette,
        comparison_path
    )
    
    print(f"Saved comparison to {comparison_path}")

def example_remove_background(input_image, output_dir):
    """Example of removing background from an image."""
    print("\n=== Removing Background ===")
    
    processor = PixelProcessor(input_image)
    
    # Remove background using different methods
    methods = ['color', 'contrast', 'edge']
    
    for method in methods:
        print(f"Removing background using {method} method...")
        result = processor.remove_background(method=method)
        
        # Save result
        output_path = os.path.join(output_dir, f"transparent_{method}.png")
        result.save(output_path)
        print(f"Saved to {output_path}")

def example_resize(input_image, output_dir):
    """Example of resizing an image with various methods."""
    print("\n=== Resizing Image ===")
    
    processor = PixelProcessor(input_image)
    
    # Resize with different methods
    methods = ['pixel-perfect', 'nearest', 'lanczos']
    
    for method in methods:
        print(f"Resizing with {method} method...")
        
        # Double the size
        result = processor.resize(scale=2.0, method=method)
        
        # Save result
        output_path = os.path.join(output_dir, f"resize_{method}.png")
        result.save(output_path)
        print(f"Saved to {output_path}")
    
    # Use specialized pixel art upscalers
    print("Applying specialized pixel art upscaling...")
    
    from pixelprocessing.effects.pixel_art_effects import upscale_pixel_art
    
    for method in ['hq2x', 'eagle']:
        print(f"Upscaling with {method} method...")
        
        # Apply upscaling
        image_array = np.array(processor.image)
        result_array = upscale_pixel_art(image_array, scale=2, method=method)
        
        # Convert to PIL and save
        result = Image.fromarray(result_array)
        output_path = os.path.join(output_dir, f"upscale_{method}.png")
        result.save(output_path)
        print(f"Saved to {output_path}")

def example_crt_effect(input_image, output_dir):
    """Example of applying CRT effect."""
    print("\n=== Applying CRT Effect ===")
    
    processor = PixelProcessor(input_image)
    
    # Apply CRT effect
    result = processor.apply_crt_effect(
        scanline_strength=0.2,
        curvature=0.1
    )
    
    # Save result
    output_path = os.path.join(output_dir, "crt_effect.png")
    result.save(output_path)
    print(f"Saved to {output_path}")

def example_palette_cycling(input_image, output_dir, palette):
    """Example of creating a palette cycling animation."""
    print("\n=== Creating Palette Cycling Animation ===")
    
    processor = PixelProcessor(input_image)
    
    # Create palette cycling animation
    animated = processor.create_palette_cycling_animation(
        palette,
        cycle_range=(0, min(8, len(palette))),
        frames=12,
        duration=100
    )
    
    # Save animation
    output_path = os.path.join(output_dir, "palette_cycling.gif")
    animated.save(output_path)
    print(f"Saved animation to {output_path}")

def example_pixelate(input_image, output_dir):
    """Example of applying pixelation effect."""
    print("\n=== Applying Pixelation Effect ===")
    
    processor = PixelProcessor(input_image)
    
    # Apply pixelation with various pixel sizes
    for pixel_size in [4, 8, 16]:
        print(f"Pixelating with {pixel_size}px blocks...")
        
        from pixelprocessing.effects.pixel_art_effects import apply_pixelation_effect
        
        # Apply pixelation
        image_array = np.array(processor.image)
        result_array = apply_pixelation_effect(
            image_array,
            pixel_size=pixel_size,
            sharpen=True,
            enhance_contrast=True
        )
        
        # Convert to PIL and save
        result = Image.fromarray(result_array)
        output_path = os.path.join(output_dir, f"pixelate_{pixel_size}.png")
        result.save(output_path)
        print(f"Saved to {output_path}")

def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description='PixelProcessing library examples')
    parser.add_argument('input_image', help='Path to input image')
    parser.add_argument('--output-dir', '-o', default='output',
                        help='Output directory for processed images')
    parser.add_argument('--example', '-e', choices=[
        'all', 'palette', 'dither', 'background', 
        'resize', 'crt', 'cycling', 'pixelate'
    ], default='all', help='Specific example to run')
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.isfile(args.input_image):
        print(f"Error: Input image {args.input_image} not found")
        sys.exit(1)
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Extract palette (needed for several examples)
    palette = None
    if args.example in ['all', 'palette', 'dither', 'cycling']:
        palette = example_extract_palette(args.input_image, args.output_dir)
    
    # Run selected example
    if args.example == 'all' or args.example == 'dither':
        example_apply_dithering(args.input_image, args.output_dir, palette)
    
    if args.example == 'all' or args.example == 'background':
        example_remove_background(args.input_image, args.output_dir)
    
    if args.example == 'all' or args.example == 'resize':
        example_resize(args.input_image, args.output_dir)
    
    if args.example == 'all' or args.example == 'crt':
        example_crt_effect(args.input_image, args.output_dir)
    
    if args.example == 'all' or args.example == 'cycling':
        example_palette_cycling(args.input_image, args.output_dir, palette)
    
    if args.example == 'all' or args.example == 'pixelate':
        example_pixelate(args.input_image, args.output_dir)
    
    print("\nAll examples completed!")
    print(f"Output files saved to {args.output_dir}/")

if __name__ == '__main__':
    main()
