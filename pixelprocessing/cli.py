import os
import sys
import click
import logging
import numpy as np
from PIL import Image
from .pixel_processor import PixelProcessor
from .core.config import Config

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """
    Pixel Processing - A tool for advanced pixel art processing.
    
    This tool provides various commands for palette extraction,
    dithering, background removal, and special effects for pixel art.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Initialize context object with config
    ctx.ensure_object(dict)
    ctx.obj['config'] = Config()
    ctx.obj['verbose'] = verbose

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--colors', '-c', type=int, default=8, help='Number of colors to extract')
@click.option('--width', '-w', type=int, default=600, help='Width of palette image')
@click.option('--height', '-h', type=int, default=50, help='Height of palette image')
@click.option('--method', '-m', type=click.Choice(['kmeans', 'median-cut']), 
              default='kmeans', help='Color extraction method')
@click.pass_context
def extract_palette(ctx, input_path, output_path, colors, width, height, method):
    """
    Extract a color palette from an image.
    
    This command analyzes the input image and generates a palette image
    containing the most dominant colors found in the image.
    """
    try:
        # Update config
        config = ctx.obj['config']
        config.num_dominant_colors = colors
        
        # Create processor
        processor = PixelProcessor(input_path)
        
        # Extract palette
        palette = processor.extract_palette(colors, method)
        
        # Save palette image
        from .strategies.color.palette_extractor import PaletteExtractor
        palette_extractor = PaletteExtractor(config)
        
        success = palette_extractor.save_palette_image(palette, output_path, width, height)
        
        if success:
            click.echo(f"✓ Extracted {len(palette)} colors to {output_path}")
        else:
            click.echo("✗ Failed to save palette image", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--colors', '-c', type=int, default=8, help='Number of colors in palette')
@click.option('--dither', '-d', 
              type=click.Choice(['none', 'floyd-steinberg', 'bayer', 'blue-noise']),
              default='none', help='Dithering algorithm')
@click.option('--strength', '-s', type=float, default=1.0, 
              help='Dithering strength (0.0-1.0)')
@click.option('--palette', '-p', type=click.Path(exists=True),
              help='Use existing palette image instead of generating one')
@click.pass_context
def dither(ctx, input_path, output_path, colors, dither, strength, palette):
    """
    Apply dithering to an image with color reduction.
    
    This command reduces the colors in an image to the specified number
    and applies dithering to create a smoother appearance.
    """
    try:
        # Create processor
        processor = PixelProcessor(input_path)
        
        # Get palette
        if palette:
            # Load palette from file
            palette_img = Image.open(palette).convert("RGB")
            palette_array = np.array(palette_img)
            
            # Extract unique colors
            unique_colors = np.unique(palette_array.reshape(-1, 3), axis=0)
            
            click.echo(f"Loaded {len(unique_colors)} colors from palette")
        else:
            # Extract palette from image
            unique_colors = processor.extract_palette(colors)
            click.echo(f"Extracted {len(unique_colors)} colors from image")
        
        # Apply dithering
        result = processor.apply_palette(unique_colors, dither, strength)
        
        # Save result
        result.save(output_path)
        click.echo(f"✓ Saved dithered image to {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--method', '-m', 
              type=click.Choice(['color', 'contrast', 'edge']),
              default='color', help='Background removal method')
@click.option('--threshold', '-t', type=int, default=30, 
              help='Threshold for background detection')
@click.option('--feather-edges', '-f', is_flag=True, 
              help='Apply feathering to edges')
@click.option('--bg-color', type=str, 
              help='Background color (R,G,B format, e.g. "255,0,0")')
@click.pass_context
def remove_bg(ctx, input_path, output_path, method, threshold, feather_edges, bg_color):
    """
    Remove the background from an image.
    
    This command detects and removes the background from an image,
    creating a transparent PNG with the foreground elements.
    """
    try:
        # Parse background color if provided
        bg_color_array = None
        if bg_color:
            try:
                bg_color_array = np.array([int(x) for x in bg_color.split(',')], dtype=np.uint8)
                if len(bg_color_array) != 3:
                    raise ValueError("Background color must have 3 components (R,G,B)")
            except Exception as e:
                click.echo(f"✗ Invalid background color format: {str(e)}", err=True)
                sys.exit(1)
        
        # Create processor
        processor = PixelProcessor(input_path)
        
        # Remove background
        result = processor.remove_background(
            method=method,
            threshold=threshold,
            feather_edges=feather_edges,
            bg_color=bg_color_array
        )
        
        # Save result
        result.save(output_path)
        click.echo(f"✓ Saved transparent image to {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--width', '-w', type=int, help='Output width in pixels')
@click.option('--height', '-h', type=int, help='Output height in pixels')
@click.option('--scale', '-s', type=float, help='Scale factor (e.g., 2.0 for 2x size)')
@click.option('--method', '-m', 
              type=click.Choice(['pixel-perfect', 'nearest', 'lanczos', 'hq2x', 'eagle']),
              default='pixel-perfect', help='Resize method')
@click.pass_context
def resize(ctx, input_path, output_path, width, height, scale, method):
    """
    Resize an image using various methods.
    
    This command resizes an image using methods optimized for pixel art,
    such as nearest neighbor (pixel-perfect) scaling or HQ2X.
    """
    try:
        # Create processor
        processor = PixelProcessor(input_path)
        
        # Resize image
        if method in ['hq2x', 'eagle']:
            # Use specialized pixel art upscaler
            from .effects.pixel_art_effects import upscale_pixel_art
            image_array = np.array(processor.image)
            
            if scale is None:
                scale = 2  # These methods only support 2x scaling
            else:
                if scale != 2:
                    click.echo(f"Warning: {method} only supports 2x scaling, using scale=2")
                scale = 2
                
            result_array = upscale_pixel_art(image_array, scale, method)
            result = Image.fromarray(result_array)
        else:
            # Use standard resize method
            result = processor.resize(width, height, scale, method)
        
        # Save result
        result.save(output_path)
        
        # Display information
        width, height = result.size
        click.echo(f"✓ Saved resized image ({width}x{height}) to {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--scanline-strength', type=float, default=0.2, 
              help='Strength of scanline effect (0.0-1.0)')
@click.option('--curvature', type=float, default=0.1, 
              help='Screen curvature amount (0.0-0.5)')
@click.option('--color-bleed', type=float, default=0.5, 
              help='Amount of color bleeding (0.0-1.0)')
@click.option('--glow', type=float, default=0.2, 
              help='Strength of glow effect (0.0-1.0)')
@click.option('--noise', type=float, default=0.05, 
              help='Amount of CRT noise (0.0-0.5)')
@click.pass_context
def crt_effect(ctx, input_path, output_path, scanline_strength, curvature, 
               color_bleed, glow, noise):
    """
    Apply a CRT monitor effect to an image.
    
    This command applies a retro CRT screen effect to an image,
    including scanlines, screen curvature, and color bleeding.
    """
    try:
        # Create processor
        processor = PixelProcessor(input_path)
        
        # Apply CRT effect
        from .effects.crt_effect import apply_crt_effect
        
        image_array = np.array(processor.image)
        result_array = apply_crt_effect(
            image_array,
            scanline_strength=scanline_strength,
            curvature=curvature,
            color_bleed=color_bleed,
            glow=glow,
            noise=noise
        )
        
        # Convert to PIL and save
        result = Image.fromarray(result_array)
        result.save(output_path)
        
        click.echo(f"✓ Saved CRT effect image to {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--colors', '-c', type=int, default=8, help='Number of colors in palette')
@click.option('--cycle-start', type=int, default=0, help='Start index for cycling')
@click.option('--cycle-end', type=int, default=8, help='End index for cycling')
@click.option('--frames', '-f', type=int, default=12, help='Number of animation frames')
@click.option('--duration', '-d', type=int, default=100, 
              help='Frame duration in milliseconds')
@click.option('--loop', '-l', type=int, default=0, 
              help='Number of loops (0 = infinite)')
@click.pass_context
def palette_cycling(ctx, input_path, output_path, colors, cycle_start, 
                    cycle_end, frames, duration, loop):
    """
    Create a palette cycling animation.
    
    This command creates an animated GIF that cycles through colors
    in the palette to create a motion effect.
    """
    try:
        # Create processor
        processor = PixelProcessor(input_path)
        
        # Extract palette
        palette = processor.extract_palette(colors)
        
        # Create palette cycling animation
        result = processor.create_palette_cycling_animation(
            palette,
            cycle_range=(cycle_start, cycle_end),
            frames=frames,
            duration=duration
        )
        
        # Save result
        result.save(output_path)
        
        click.echo(f"✓ Saved palette cycling animation to {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--pixel-size', '-p', type=int, default=8, 
              help='Size of pixelation blocks')
@click.option('--sharpen/--no-sharpen', default=True, 
              help='Apply sharpening')
@click.option('--enhance-contrast/--no-enhance-contrast', default=True, 
              help='Enhance contrast')
@click.pass_context
def pixelate(ctx, input_path, output_path, pixel_size, sharpen, enhance_contrast):
    """
    Apply a pixelation effect to an image.
    
    This command creates a pixel art effect by reducing the resolution
    and applying optional sharpening and contrast enhancement.
    """
    try:
        # Create processor
        processor = PixelProcessor(input_path)
        
        # Apply pixelation effect
        from .effects.pixel_art_effects import apply_pixelation_effect
        
        image_array = np.array(processor.image)
        result_array = apply_pixelation_effect(
            image_array,
            pixel_size=pixel_size,
            sharpen=sharpen,
            enhance_contrast=enhance_contrast
        )
        
        # Convert to PIL and save
        result = Image.fromarray(result_array)
        result.save(output_path)
        
        click.echo(f"✓ Saved pixelated image to {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--frame-width', '-w', type=int, required=True, 
              help='Width of each frame in pixels')
@click.option('--frame-height', '-h', type=int, required=True, 
              help='Height of each frame in pixels')
@click.option('--duration', '-d', type=int, default=100, 
              help='Frame duration in milliseconds')
@click.option('--loop', '-l', type=int, default=0, 
              help='Number of loops (0 = infinite)')
@click.option('--ping-pong', '-p', is_flag=True, 
              help='Create ping-pong animation (forward and backward)')
@click.pass_context
def sprite_animation(ctx, input_path, output_path, frame_width, frame_height, 
                     duration, loop, ping_pong):
    """
    Create an animation from a sprite sheet.
    
    This command extracts frames from a sprite sheet and creates
    an animated GIF.
    """
    try:
        # Create processor (just to use Image loading)
        processor = PixelProcessor(input_path)
        
        # Get the sprite sheet image
        sprite_sheet = processor.image
        
        # Create animation from sprite sheet
        from .effects.animation_effects import create_sprite_sheet_animation, create_ping_pong_animation
        
        # Extract frames
        frames = create_sprite_sheet_animation(
            sprite_sheet,
            frame_width,
            frame_height
        )
        
        # Apply ping-pong if requested
        if ping_pong and frames:
            frames = create_ping_pong_animation(frames)
        
        # Save animation
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                optimize=True,
                duration=duration,
                loop=loop
            )
            
            click.echo(f"✓ Saved animation with {len(frames)} frames to {output_path}")
        else:
            click.echo("✗ No frames extracted from sprite sheet", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"✗ Error: {str(e)}", err=True)
        if ctx.obj['verbose']:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

def main():
    """
    Main entry point for the CLI.
    """
    cli(obj={})

if __name__ == '__main__':
    main()
