# Pixel Processing

A Python utility for advanced pixel art processing, including palette reduction, dithering, and special effects.

## Features

- **Palette extraction and mapping**: Create optimized color palettes from images
- **Multiple dithering algorithms**: Floyd-Steinberg, Bayer, Blue Noise
- **Background removal**: Extract foreground elements with transparency
- **Special effects**: CRT simulation, pixel art scaling, and more

## Installation

```bash
pip install pixelprocessing
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/codedevmk/python-pixel-processing.git
```

## Quick Start

```python
from pixelprocessing import PixelProcessor
from pixelprocessing.strategies import FloydSteinbergDither, PaletteExtractor

# Load an image and apply processing
processor = PixelProcessor('input.png')

# Extract an 8-color palette
palette = processor.extract_palette(colors=8)

# Apply dithering with the palette
processed = processor.apply_dithering(palette, method='floyd-steinberg')

# Save the result
processed.save('output.png')
```

## Command Line Usage

```bash
# Apply dithering to an image
pixelprocessing dither input.png output.png --colors 8 --method floyd-steinberg

# Remove background
pixelprocessing bg-remove input.png output.png --method color

# Extract a palette from an image
pixelprocessing extract-palette input.png palette.png --colors 16
```

## Documentation

See the [documentation](https://github.com/codedevmk/python-pixel-processing/wiki) for more details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
