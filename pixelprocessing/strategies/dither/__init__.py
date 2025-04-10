# Dithering-related strategy implementations

from .dither_manager import DitherManager
from .floyd_steinberg import FloydSteinbergDither
from .bayer import BayerDither
from .blue_noise import BlueNoiseDither

__all__ = [
    'DitherManager',
    'FloydSteinbergDither',
    'BayerDither',
    'BlueNoiseDither'
]
